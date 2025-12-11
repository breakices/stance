#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
五分类训练 + 多视角测试脚本

训练：5 分类（s_against / against / stance_not_inferrable / favor / s_favor）

测试阶段输出：
1) 5-class 结果（整体 + 按 topic 分组 + topic 宏平均）
2) 3-class 结果（反对/中立/支持）
   - 反对：s_against, against
   - 中立：stance_not_inferrable
   - 支持：favor, s_favor
3) 2-class 结果（支持 vs 反对）
   - 丢掉 GT 为中立的样本（stance_not_inferrable）
   - 预测为中立一律算错（映射为“与 GT 相反”的类）
   - 同样按 topic 分组 + topic 宏平均
"""

import os
import random
from typing import Dict, Any, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# =====================
# 配置区：按需修改
# =====================

MODEL_NAME = "/online1/sc100123/sc100123/agentic_moderation/qwen3-4b-Ins/"
TRAIN_PATH = "/online1/sc100123/sc100123/agentic_moderation/LLaMA-Factory/sd/data/train.parquet"
TEST_PATH  = "/online1/sc100123/sc100123/agentic_moderation/LLaMA-Factory/sd/data/test.parquet"

TEXT_COL = "complete_prompt"   # 输入文本列
RAW_LABEL_COL = "label"        # 原始字符串标签列
TOPIC_COL = "topic"            # topic 列
EXCLUDE_RAW_LABELS = {"undecided"}

# 5-class 映射：与系统 prompt 中 digit↔label 对齐
LABEL2ID_5 = {
    "s_against": 0,
    "against": 1,
    "stance_not_inferrable": 2,
    "favor": 3,
    "s_favor": 4,
}
NUM_LABELS_5 = 5

# 5 -> 3 映射（反对/中立/支持）
# index: 0 1 2 3 4  (对应上面 5-class id)
# value: 0 0 1 2 2  (0=反对, 1=中立, 2=支持)
MAP_5_TO_3 = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
NUM_LABELS_3 = 3

MAX_LENGTH = 20000
BATCH_SIZE = 2           # 显存不够可以再调小
NUM_EPOCHS = 5
BACKBONE_LR = 5e-6
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
SEED = 42
SAVE_PATH = "qwen3_5cls_full_finetune.pt"


# =====================
# 工具函数
# =====================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_confusion_metrics(y_true: torch.Tensor,
                              y_pred: torch.Tensor,
                              num_labels: int) -> Dict[str, float]:
    """
    给定 y_true / y_pred（int tensor），计算：
      - accuracy
      - 宏平均 precision / recall / F1
    """
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


# =====================
# Dataset：5 分类训练用
# =====================

class StanceParquetDataset(Dataset):
    """
    从 parquet 读取数据，做 5 分类训练与测试。
    返回:
      - input_ids, attention_mask, labels (5-class id)
      - topic (字符串，用于按 topic 评估)
    """

    def __init__(self, parquet_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        assert os.path.exists(parquet_path), f"{parquet_path} 不存在"

        df = pd.read_parquet(parquet_path)

        # 过滤 undecided
        if RAW_LABEL_COL not in df.columns:
            raise ValueError(f"{parquet_path} 中找不到列 '{RAW_LABEL_COL}'")
        if TOPIC_COL not in df.columns:
            raise ValueError(f"{parquet_path} 中找不到列 '{TOPIC_COL}'")

        before = len(df)
        df = df[~df[RAW_LABEL_COL].isin(EXCLUDE_RAW_LABELS)].reset_index(drop=True)
        dropped = before - len(df)
        print(f"[Dataset] {parquet_path}: drop {dropped} rows with label in {EXCLUDE_RAW_LABELS}")

        unknown_labels = set(df[RAW_LABEL_COL].unique()) - set(LABEL2ID_5.keys())
        if unknown_labels:
            raise ValueError(f"发现未知标签: {unknown_labels}，请更新 LABEL2ID_5 或 EXCLUDE_RAW_LABELS。")

        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = NUM_LABELS_5

        if len(self.df) == 0:
            raise ValueError(f"After filtering, no samples left in {parquet_path}")

        print(f"[Dataset] {parquet_path}: {len(self.df)} samples, num_labels={self.num_labels}")

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
            "labels": torch.tensor(label_id, dtype=torch.long),
            "topic": topic,
        }


# =====================
# 模型：Qwen3 + 5 分类头
# =====================

class Qwen3StanceClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 5, pool_type: str = "last"):
        super().__init__()
        self.num_labels = num_labels
        self.pool_type = pool_type

        # GPU 用 bfloat16，CPU 用 float32
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        hidden_size = self.backbone.config.hidden_size

        backbone_dtype = next(self.backbone.parameters()).dtype
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=backbone_dtype)

        # 可选：gradient checkpointing 省显存
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def pool(self, last_hidden_state, attention_mask):
        if self.pool_type == "last":
            last_indices = attention_mask.sum(dim=1) - 1
            last_indices = last_indices.clamp(min=0)
            pooled = last_hidden_state[
                torch.arange(last_hidden_state.size(0), device=last_hidden_state.device),
                last_indices,
            ]
        elif self.pool_type == "mean":
            mask = attention_mask.unsqueeze(-1)
            masked_hidden = last_hidden_state * mask
            sum_hidden = masked_hidden.sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / lengths
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        return pooled

    def forward(self, input_ids, attention_mask=None, labels=None):
        core = getattr(self.backbone, "model", self.backbone)
        outputs = core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state

        pooled = self.pool(last_hidden_state, attention_mask)
        pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.float(), labels)

        return {"logits": logits, "loss": loss}


# =====================
# 训练
# =====================

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(dataloader, desc="Train")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # 'topic' 不需要 to(device)，直接忽略

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

        avg_loss = total_loss / total_count
        avg_acc = total_correct / total_count
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})

    return total_loss / total_count, total_correct / total_count


# =====================
# 测试：一次性算 5-class / 3-class / 2-class，多 topic 分组
# =====================

@torch.no_grad()
def evaluate_multi_view(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_true5: List[int] = []
    all_pred5: List[int] = []
    all_topics: List[str] = []

    pbar = tqdm(dataloader, desc="Eval")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        topics = batch["topic"]  # list[str]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        preds = logits.argmax(dim=-1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        all_true5.extend(labels.cpu().tolist())
        all_pred5.extend(preds.cpu().tolist())
        all_topics.extend(list(topics))

        avg_loss = total_loss / total_count
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    if total_count == 0:
        print("[Eval] No samples in dataloader.")
        return

    true5 = torch.tensor(all_true5, dtype=torch.long)
    pred5 = torch.tensor(all_pred5, dtype=torch.long)

    # 3-class 映射
    true3 = MAP_5_TO_3[true5]
    pred3 = MAP_5_TO_3[pred5]

    print("\n===== Overall metrics (whole test set) =====")

    # ---- 5-class overall ----
    metrics_5 = compute_confusion_metrics(true5, pred5, NUM_LABELS_5)
    print("[5-class] overall: "
          f"acc={metrics_5['accuracy']:.4f}, "
          f"P_macro={metrics_5['macro_precision']:.4f}, "
          f"R_macro={metrics_5['macro_recall']:.4f}, "
          f"F1_macro={metrics_5['macro_f1']:.4f}")

    # ---- 3-class overall ----
    metrics_3 = compute_confusion_metrics(true3, pred3, NUM_LABELS_3)
    print("[3-class] overall (oppose/neutral/support): "
          f"acc={metrics_3['accuracy']:.4f}, "
          f"P_macro={metrics_3['macro_precision']:.4f}, "
          f"R_macro={metrics_3['macro_recall']:.4f}, "
          f"F1_macro={metrics_3['macro_f1']:.4f}")

    # ---- 2-class overall（支持 vs 反对，GT 非中立，预测中立算错）----
    # 先过滤掉 GT 中立 (true3 == 1)
    mask_non_neutral = (true3 != 1)
    true3_eff = true3[mask_non_neutral]
    pred3_eff = pred3[mask_non_neutral]

    if true3_eff.numel() > 0:
        # true2: 0 = oppose, 1 = support
        true2 = torch.where(true3_eff == 0,
                            torch.zeros_like(true3_eff),
                            torch.ones_like(true3_eff))

        pred2 = torch.empty_like(true2)
        for i in range(true2.numel()):
            if pred3_eff[i] == 0:
                pred2[i] = 0
            elif pred3_eff[i] == 2:
                pred2[i] = 1
            else:
                # pred3 == 1 (neutral) -> 强制算错：映射成与 GT 相反的类别
                pred2[i] = 1 - true2[i]

        metrics_2 = compute_confusion_metrics(true2, pred2, num_labels=2)
        print("[2-class] overall (support vs oppose, GT non-neutral, neutral pred = wrong): "
              f"acc={metrics_2['accuracy']:.4f}, "
              f"P_macro={metrics_2['macro_precision']:.4f}, "
              f"R_macro={metrics_2['macro_recall']:.4f}, "
              f"F1_macro={metrics_2['macro_f1']:.4f}")
    else:
        print("[2-class] overall: no non-neutral GT samples; metrics not computed.")
        metrics_2 = None

    # ========= 按 topic 分组 =========

    print("\n===== Per-topic metrics =====")

    topic2indices: Dict[str, List[int]] = {}
    for idx, t in enumerate(all_topics):
        topic2indices.setdefault(t, []).append(idx)

    # 记录 topic-level 指标，用于 macro over topics
    topic_metrics_5 = []
    topic_metrics_3 = []
    topic_metrics_2 = []

    for topic, indices in sorted(topic2indices.items(), key=lambda x: x[0]):
        idx_tensor = torch.tensor(indices, dtype=torch.long)

        t5 = true5[idx_tensor]
        p5 = pred5[idx_tensor]
        t3 = true3[idx_tensor]
        p3 = pred3[idx_tensor]

        m5_t = compute_confusion_metrics(t5, p5, NUM_LABELS_5)
        m3_t = compute_confusion_metrics(t3, p3, NUM_LABELS_3)

        # 2-class: 同样先过滤掉 GT 中立
        mask_non_neutral_t = (t3 != 1)
        t3_eff_t = t3[mask_non_neutral_t]
        p3_eff_t = p3[mask_non_neutral_t]

        if t3_eff_t.numel() > 0:
            true2_t = torch.where(t3_eff_t == 0,
                                  torch.zeros_like(t3_eff_t),
                                  torch.ones_like(t3_eff_t))
            pred2_t = torch.empty_like(true2_t)
            for i in range(true2_t.numel()):
                if p3_eff_t[i] == 0:
                    pred2_t[i] = 0
                elif p3_eff_t[i] == 2:
                    pred2_t[i] = 1
                else:
                    pred2_t[i] = 1 - true2_t[i]
            m2_t = compute_confusion_metrics(true2_t, pred2_t, num_labels=2)
            topic_metrics_2.append(m2_t)
        else:
            m2_t = None  # 该 topic 下没有非中立样本

        topic_metrics_5.append(m5_t)
        topic_metrics_3.append(m3_t)

        print(f"\n--- Topic: {topic} ---")
        print(f"[5-class]  acc={m5_t['accuracy']:.4f}, "
              f"P_macro={m5_t['macro_precision']:.4f}, "
              f"R_macro={m5_t['macro_recall']:.4f}, "
              f"F1_macro={m5_t['macro_f1']:.4f}")
        print(f"[3-class]  acc={m3_t['accuracy']:.4f}, "
              f"P_macro={m3_t['macro_precision']:.4f}, "
              f"R_macro={m3_t['macro_recall']:.4f}, "
              f"F1_macro={m3_t['macro_f1']:.4f}")
        if m2_t is not None:
            print(f"[2-class]  acc={m2_t['accuracy']:.4f}, "
                  f"P_macro={m2_t['macro_precision']:.4f}, "
                  f"R_macro={m2_t['macro_recall']:.4f}, "
                  f"F1_macro={m2_t['macro_f1']:.4f}")
        else:
            print("[2-class]  (no non-neutral GT samples for this topic)")

    # ========= 按 topic 宏平均 =========

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

    avg_5 = average_topic_metrics(topic_metrics_5)
    avg_3 = average_topic_metrics(topic_metrics_3)
    avg_2 = average_topic_metrics(topic_metrics_2)

    print("\n===== Topic-macro-averaged metrics =====")
    print("[5-class] topic-avg: "
          f"acc={avg_5['accuracy']:.4f}, "
          f"P_macro={avg_5['macro_precision']:.4f}, "
          f"R_macro={avg_5['macro_recall']:.4f}, "
          f"F1_macro={avg_5['macro_f1']:.4f}")
    print("[3-class] topic-avg: "
          f"acc={avg_3['accuracy']:.4f}, "
          f"P_macro={avg_3['macro_precision']:.4f}, "
          f"R_macro={avg_3['macro_recall']:.4f}, "
          f"F1_macro={avg_3['macro_f1']:.4f}")
    print("[2-class] topic-avg (only topics with non-neutral GT): "
          f"acc={avg_2['accuracy']:.4f}, "
          f"P_macro={avg_2['macro_precision']:.4f}, "
          f"R_macro={avg_2['macro_recall']:.4f}, "
          f"F1_macro={avg_2['macro_f1']:.4f}")


# =====================
# main
# =====================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = StanceParquetDataset(TRAIN_PATH, tokenizer, max_length=MAX_LENGTH)
    test_dataset  = StanceParquetDataset(TEST_PATH,  tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = Qwen3StanceClassifier(MODEL_NAME, num_labels=NUM_LABELS_5, pool_type="last")
    model.to(device)

    print(f"[Info] Finetune backbone (lr={BACKBONE_LR}) and classifier head (lr={HEAD_LR}).")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(),   "lr": BACKBONE_LR},
            {"params": model.classifier.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"[Train] loss={train_loss:.4f}, acc={train_acc:.4f}")

    print("\n===== Final evaluation on TEST set (5/3/2-class, per-topic) =====")
    evaluate_multi_view(model, test_loader, device)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"[Info] Saved model weights to {SAVE_PATH}")


if __name__ == "__main__":
    main()
