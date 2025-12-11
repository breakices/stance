import json
import argparse
import re
from typing import Optional, Any


def extract_pred_digit(cot: str) -> Optional[int]:
    """
    Extract the prediction digit (0-4) from the model output string.
    It expects the final answer to appear after </reasoning>.
    """
    if not isinstance(cot, str):
        return None

    # Find the closing reasoning tag
    tag = "</reasoning>"
    idx = cot.lower().rfind(tag)
    if idx != -1:
        tail = cot[idx + len(tag):]
    else:
        # Fallback: use the whole string if the tag is missing
        tail = cot

    # Strip whitespace
    tail = tail.strip()

    # Option 1: first digit in tail
    for ch in tail:
        if ch in "01234":
            return int(ch)

    # Option 2: regex as backup (in case of noise)
    m = re.search(r"[0-4]", tail)
    if m:
        return int(m.group(0))

    return None


def gold_label_to_digit(label: Any) -> Optional[int]:
    """
    Map the gold label to a digit 0-4.
    Supports both numeric and string labels like 'stance_not_inferrable'.
    """
    # Already an int
    if isinstance(label, int):
        return label if 0 <= label <= 4 else None

    # Could be a numeric string
    if isinstance(label, str):
        s = label.strip()
        if s in {"0", "1", "2", "3", "4"}:
            return int(s)

        # Map textual labels to digits
        mapping = {
            "s_against": 0,
            "against": 1,
            "stance_not_inferrable": 2,
            "stance_not_inferable": 2,  # just in case of spelling variants
            "favor": 3,
            "s_favor": 4,
        }
        key = s.lower()
        if key in mapping:
            return mapping[key]

    return None


def evaluate(pred_json_path: str) -> None:
    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_total = 0
    n_valid = 0
    n_correct = 0
    n_pred_none = 0
    n_gold_none = 0

    for item in data:
        n_total += 1

        label = item.get("label", None)
        cot = item.get("cot", "")

        gold = gold_label_to_digit(label)
        pred = extract_pred_digit(cot)

        if gold is None:
            n_gold_none += 1
        if pred is None:
            n_pred_none += 1

        if gold is None or pred is None:
            continue

        n_valid += 1
        if gold == pred:
            n_correct += 1

    print(f"Total samples in file: {n_total}")
    print(f"Valid samples used for accuracy (both gold and pred available): {n_valid}")
    print(f"Samples with missing/invalid gold label: {n_gold_none}")
    print(f"Samples with missing/invalid prediction: {n_pred_none}")

    if n_valid > 0:
        acc = n_correct / n_valid
        print(f"Accuracy: {acc:.4f}")
    else:
        print("No valid samples to evaluate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path to the JSON file with predictions (unit_id, target_text, label, cot)."
    )
    args = parser.parse_args()
    evaluate(args.pred_path)
