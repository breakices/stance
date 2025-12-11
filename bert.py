#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from typing import Dict, Any, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

ENCODER_MODEL = "./roberta"

TRAIN_PATH = "/online1/sc100123/sc100123/agentic_moderation/LLaMA-Factory/sd/data/train.parquet"
TEST_PATH  = "/online1/sc100123/sc100123/agentic_moderation/LLaMA-Factory/sd/data/test.parquet"

TEXT_COL = "complete_prompt"
RAW_LABEL_COL = "label"
TOPIC_COL = "topic"
EXCLUDE_RAW_LABELS = {"undecided"}

LABEL2ID_5 = {
    "s_against": 0,
    "against": 1,
    "stance_not_inferrable": 2,
    "favor": 3,
    "s_favor": 4,
}
NUM_LABELS_5 = 5

MAP_5_TO_3 = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
NUM_LABELS_3 = 3

MAX_LENGTH = 512
BATCH_SIZE = 8
NUM_EPOCHS = 5
LR_ENCODER = 2e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.01
SEED = 42
SAVE_PATH = "hier_roberta_stance.pt"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_confusion_metrics(y_true: torch.Tensor,
                              y_pred: torch.Tensor,
                              num_labels: int) -> Dict[str, float]:
    cm = torch.zeros((num_labels, num_labels), dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        if 0 <= int(t) < num_labels and 0 <= int(p) < num_labels:
            cm[int(t), int(p)] += 1

    total = cm.sum().item()
    if total == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    accuracy = cm.diag().sum().item() / total

    tp = cm.diag().float()
    pred_sum = cm.sum(dim=0).float()
    true_sum = cm.sum(dim=1).float()

    precision_per_class = tp / pred_sum.clamp(min=1.0)
    recall_per_class = tp / true_sum.clamp(min=1.0)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class).clamp(min=1e-12)

    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def average_topic_metrics(topic_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not topic_metrics:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    avg = {k: 0.0 for k in keys}
    for m in topic_metrics:
        for k in keys:
            avg[k] += m[k]
    for k in keys:
        avg[k] /= len(topic_metrics)
    return avg


class StanceParquetDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        assert os.path.exists(parquet_path), f"{parquet_path} 不存在"

        df = pd.read_parquet(parquet_path)

        if RAW_LABEL_COL not in df.columns:
            raise ValueError(f"{parquet_path} 中找不到列 '{RAW_LABEL_COL}'")
        if TOPIC_COL not in df.columns:
            raise ValueError(f"{parquet_path} 中找不到列 '{TOPIC_COL}'")

        before = len(df)
        df = df[~df[RAW_LABEL_COL].isin(EXCLUDE_RAW_LABELS)].reset_index(drop=True)
        dropped = before - len(df)
        print(f"[Dataset] {parquet_path}: drop {dropped} rows with label in {EXCLUDE_RAW_LABELS}")

        unknown = set(df[RAW_LABEL_COL].unique()) - set(LABEL2ID_5.keys())
        if unknown:
            raise ValueError(f"发现未知标签: {unknown}，请更新 LABEL2ID_5 或 EXCLUDE_RAW_LABELS。")

        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

        if len(self.df) == 0:
            raise ValueError(f"After filtering, no samples left in {parquet_path}")

        print(f"[Dataset] {parquet_path}: {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        text = row[TEXT_COL]
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)

        raw_label = row[RAW_LABEL_COL]
        label_id = LABEL2ID_5[raw_label]

        topic = row[TOPIC_COL]
        if not isinstance(topic, str):
            topic = "" if pd.isna(topic) else str(topic)

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels_5": torch.tensor(label_id, dtype=torch.long),
            "topic": topic,
        }


class HierStanceRoberta(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.head_polarity = nn.Linear(hidden_size, 1)
        self.head_direction = nn.Linear(hidden_size, 1)
        self.head_strength = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls = outputs.last_hidden_state[:, 0, :]

        logit_polarity = self.head_polarity(cls).squeeze(-1)
        logit_direction = self.head_direction(cls).squeeze(-1)
        logit_strength = self.head_strength(cls).squeeze(-1)

        return logit_polarity, logit_direction, logit_strength


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_count = 0

    pbar = tqdm(dataloader, desc="Train")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_5 = batch["labels_5"].to(device)

        optimizer.zero_grad()

        logit_polarity, logit_direction, logit_strength = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        y_polarity = (labels_5 != 2).float()
        y_direction = torch.where(labels_5 >= 3, 1.0, 0.0)
        y_strength = torch.where(
            (labels_5 == 0) | (labels_5 == 4),
            1.0,
            0.0,
        )

        mask_non_neutral = (labels_5 != 2)

        loss = 0.0

        loss_p = bce(logit_polarity, y_polarity)
        loss += loss_p

        if mask_non_neutral.any():
            loss_d = bce(logit_direction[mask_non_neutral], y_direction[mask_non_neutral])
            loss += loss_d
        else:
            loss_d = torch.tensor(0.0, device=device)

        if mask_non_neutral.any():
            loss_s = bce(logit_strength[mask_non_neutral], y_strength[mask_non_neutral])
            loss += loss_s
        else:
            loss_s = torch.tensor(0.0, device=device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels_5.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        avg_loss = total_loss / total_count
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "Lp": f"{loss_p.item():.3f}",
            "Ld": f"{loss_d.item():.3f}",
            "Ls": f"{loss_s.item():.3f}",
        })

    return total_loss / total_count


@torch.no_grad()
def evaluate_multi_view(model, dataloader, device):
    model.eval()

    all_true5: List[int] = []
    all_pred5: List[int] = []
    all_topics: List[str] = []

    pbar = tqdm(dataloader, desc="Eval")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_5 = batch["labels_5"].to(device)
        topics = batch["topic"]

        logit_polarity, logit_direction, logit_strength = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        prob_p = torch.sigmoid(logit_polarity)
        prob_d = torch.sigmoid(logit_direction)
        prob_s = torch.sigmoid(logit_strength)

        pred_p = (prob_p >= 0.5).long()
        pred_d = (prob_d >= 0.5).long()
        pred_s = (prob_s >= 0.5).long()

        pred_5 = torch.full_like(labels_5, 2)

        mask_pol = (pred_p == 1)
        mask_favor = mask_pol & (pred_d == 1)
        mask_against = mask_pol & (pred_d == 0)

        mask_favor_strong = mask_favor & (pred_s == 1)
        mask_favor_weak   = mask_favor & (pred_s == 0)
        mask_against_strong = mask_against & (pred_s == 1)
        mask_against_weak   = mask_against & (pred_s == 0)

        pred_5[mask_favor_strong]   = 4
        pred_5[mask_favor_weak]     = 3
        pred_5[mask_again_]()_
