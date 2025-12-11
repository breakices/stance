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

TRAIN_PATH = "./LLaMA-Factory/sd/data/train.parquet"
TEST_PATH  = "./LLaMA-Factory/sd/data/test.parquet"

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
        if not isinstan
