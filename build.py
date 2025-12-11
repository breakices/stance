import os
import json
import argparse
import pandas as pd


DEFAULT_PARQUET_PATH = "/online1/sc100123/sc100123/agentic_moderation/LLaMA-Factory/sd/data/train.parquet"
DEFAULT_COT_PATH = "/online1/sc100123/sc100123/agentic_moderation/sd/cot.json"
DEFAULT_OUTPUT_PATH = "/online1/sc100123/sc100123/agentic_moderation/train.json"


FALLBACK_SYSTEM_PROMPT = """You are a careful, neutral annotator.

Your task is to classify whether the TARGET_POST_TEXT expresses a stance toward the TOPIC itself (favor / against / stance_not_inferrable). Do NOT judge whether the post agrees with, replies to, or criticises another author.

You will be given the following fields:
* TARGET_POST_TEXT — the exact post being labeled (the immediate text under annotation).
* TOP_LEVEL_POST_TEXT — the top/root post of the thread (may be empty).
* PARENT_POST_TEXT_i — ancestor comments/posts (closest parent is PARENT_POST_TEXT_1; may be empty).
* USER_PROFILE (optional) — background information about the author; use only to resolve ambiguity after using the textual context.
* TARGET_POST's TOPIC definition — including what counts as "in_favor" vs "against" for this specific TOPIC.

Digit ↔ Text mapping:
* 0 = s_against  (strongly against)
* 1 = against
* 2 = stance_not_inferrable
* 3 = favor
* 4 = s_favor  (strongly in favor)

Decision rules:
1. Polarity toward TOPIC
- Judge strictly by the TOPIC definition; treat “in_favor/against” there as the reference meaning. Do not broaden or redefine the TOPIC. If the text is off-scope, predict 2 (stance_not_inferrable).
- Output favor or against only when the post clearly supports or opposes the TOPIC itself.
- If TARGET_POST_TEXT alone is insufficient, consult TOP_LEVEL_POST_TEXT and PARENT_POST_TEXT_i for explicit or implicit statements about the TOPIC; use them only to decide stance toward the TOPIC itself (not to infer agreement/disagreement with other authors). If still ambiguous, output 2.

2. Handling not-inferrable cases
- Output 2 (stance_not_inferrable) if any of the following holds:
  (a) the text is irrelevant to the TOPIC,
  (b) the text is not understandable (e.g., incoherent/too fragmentary),
  (c) the text mainly asks for information/opinions/clarification without revealing a stance,
  (d) the text lacks sufficient clues to determine polarity even after using allowed context.
- If the evidence for favor and against is mixed or ambiguous (no clear tilt), do NOT output “undecided” or guess a side; prefer 2. Do not simulate multi-annotator vote ties.

3. Intensity (strong vs. weak)
- If polarity is decided, use the s_ prefix (s_favor / s_against) when the post shows clear intensity: explicit strong advocacy or opposition, calls to action, insults/profanity, threatening or dehumanizing language, or broad sweeping generalizations that indicate strong conviction.
  * strong → 4 (s_favor) or 0 (s_against)
  * weak/neutral strength but with polarity → 3 (favor) or 1 (against)

4. Use of USER_PROFILE (if provided)
- Use the profile only to resolve ambiguity in the TARGET_POST_TEXT after using allowed context (TOP_LEVEL / PARENT). Do not infer stance from the profile alone.
- When the profile conflicts with the target text, always follow the target text. Evidence order: TARGET_POST_TEXT > context (TOP_LEVEL / PARENT) > profile. If still ambiguous, output 2.

Output format:
First, think step by step and explain your reasoning inside a single pair of tags:

<reasoning>
...your detailed reasoning steps here...
</reasoning>

After the closing </reasoning> tag, output exactly ONE character from {0,1,2,3,4} as the final label, with:
- no extra text,
- no spaces,
- no punctuation,
- no explanations.
"""


def build_user_input_from_row(row: dict) -> str:
    """
    Build the user input string for one example.

    Priority:
    1) If the parquet has a 'user_prompt' column, use it directly (this matches the original dataset).
    2) Otherwise, construct a minimal structured prompt from basic fields.
    """
    if "user_prompt" in row and isinstance(row["user_prompt"], str):
        return row["user_prompt"]

    topic = row.get("topic", "")
    target_text = row.get("target_text", "")
    top_level = row.get("top_level_post_text", "")
    parent_posts = row.get("parent_posts", "")
    user_profile = row.get("user_profile", "")
    topic_definition = row.get("topic_definition", "")

    parts = []
    parts.append("### USER_PROFILE (auxiliary, may be empty)")
    parts.append(str(user_profile))

    parts.append("\n### TOPIC")
    parts.append(str(topic))

    parts.append("\n### TARGET_POST_TEXT")
    parts.append(str(target_text))

    parts.append("\n### TOP_LEVEL_POST_TEXT")
    parts.append(str(top_level))

    parts.append("\n### PARENT_POST_TEXT")
    parts.append(str(parent_posts))

    parts.append("\n### TARGET_POST'S TOPIC DEFINITION")
    parts.append(str(topic_definition))

    return "\n".join(parts)


def main(parquet_path: str, cot_path: str, output_path: str) -> None:
    # Load original parquet
    df = pd.read_parquet(parquet_path)
    records = df.to_dict(orient="records")

    # Build index from unit_id (string) to row dict
    index_by_id = {}
    for r in records:
        uid = r.get("unit_id")
        if uid is None:
            continue
        uid_str = str(uid)
        index_by_id[uid_str] = r

    # Determine system prompt: use parquet column if available, else fallback
    if "system_prompt" in df.columns and len(df) > 0:
        system_prompt = str(df["system_prompt"].iloc[0])
    else:
        system_prompt = FALLBACK_SYSTEM_PROMPT

    # Load COT JSON
    with open(cot_path, "r", encoding="utf-8") as f:
        cot_data = json.load(f)

    sharegpt_data = []
    missing_count = 0

    for item in cot_data:
        uid = item.get("unit_id")
        cot = item.get("cot", "")

        uid_str = str(uid) if uid is not None else None
        if uid_str is None or uid_str not in index_by_id:
            missing_count += 1
            continue

        row = index_by_id[uid_str]
        user_input = build_user_input_from_row(row)
        cot_str = str(cot)

        sharegpt_data.append(
            {
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "user", "value": user_input},
                    {"from": "assistant", "value": cot_str},
                ]
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

    print(f"Built sharegpt dataset with {len(sharegpt_data)} examples.")
    if missing_count > 0:
        print(f"Warning: {missing_count} examples in COT json had missing/non-matching unit_id and were skipped.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_path",
        type=str,
        default=DEFAULT_PARQUET_PATH,
        help="Path to the original parquet file with raw stance data.",
    )
    parser.add_argument(
        "--cot_path",
        type=str,
        default=DEFAULT_COT_PATH,
        help="Path to the JSON file with COT outputs (unit_id, cot, label, etc.).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the sharegpt-formatted dataset.",
    )
    args = parser.parse_args()
    main(args.parquet_path, args.cot_path, args.output_path)
