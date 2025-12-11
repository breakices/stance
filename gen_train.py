import os
import sys
import argparse
import json
import pandas as pd
from typing import Any, Optional

# Adjust Python path to import VLLM from utils
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import Optional, List, Dict, Any
import torch
from PIL import Image
import base64
from io import BytesIO
import re
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info 


class VLLM:
    def __init__(
            self,
            model: str,
            max_model_len: int = 32768,
            gpu_memory_utilization: float = 0.9,
            tensor_parallel_size: Optional[int] = None,
            enable_lora: bool = False,
            trust_remote_code: bool = True,
            limit_mm_per_prompt: Optional[dict] = None
    ):
        
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=enable_lora,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt=limit_mm_per_prompt
        )

        self.is_multimodal = False
        try:
            self.processor = AutoProcessor.from_pretrained(model)
            self.is_multimodal = True
        except:
            self.processor = None

        self.tokenizer = self.llm.get_tokenizer()

    def _encode_image(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()

    def _process_input(self, input_str: str) -> Dict[str, Any]:

        image_tags = re.findall(r'<image>(.*?)</image>', input_str)
        images = [self._encode_image(path) for path in image_tags]
        text = re.sub(r'<image>.*?</image>', '', input_str).strip()
        return {"text": text, "images": images, "is_multimodal": bool(images)}

    def _create_text_prompt(self, text: str, system_prompt: str, enable_thinking: bool) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

    def _create_mm_prompt(self, input_data: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        if not self.processor:
            raise ValueError("Multi-modal processing requires a valid processor")

        messages = [{"role": "system", "content": system_prompt}]
        content = []
        for img_b64 in input_data["images"]:
            content.append({"type": "image", "image": f"data:image/jpeg;base64,{img_b64}"})
        if input_data["text"]:
            content.append({"type": "text", "text": input_data["text"]})
        messages.append({"role": "user", "content": content})

        image_inputs, _ = process_vision_info(messages)
        mm_data = {"image": image_inputs} if image_inputs else {}


        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return {"prompt": text_prompt, "multi_modal_data": mm_data}

    def generate(
            self,
            inputs: List[str],
            system_prompt: str = "You are a helpful assistant.",
            enable_thinking: bool = False,
            temperature: float = 0.8,
            max_tokens: int = 10000,
            top_p: float = 0.95,
            lora_path: Optional[str] = None
    ) -> List[str]:
        processed = [self._process_input(s) for s in inputs]
        text_prompts, mm_prompts = [], []

        for data in processed:
            if data["is_multimodal"]:
                if not self.is_multimodal:
                    raise ValueError("Multi-modal input detected but model not initialized with multi-modal support.")
                mm_prompts.append(self._create_mm_prompt(data, system_prompt))
            else:
                text_prompts.append(self._create_text_prompt(data["text"], system_prompt, enable_thinking))

        results = []

        if text_prompts:
            outs = self.llm.generate(
                prompts=text_prompts,
                sampling_params=SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens),
                lora_request=LoRARequest("adapter", 1, lora_path) if lora_path else None
            )
            results.extend([o.outputs[0].text for o in outs])

        if mm_prompts and self.is_multimodal:
            prompts = [p for p in mm_prompts if p]
            outs = self.llm.generate(
                prompts,
                sampling_params=SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens),
                lora_request=LoRARequest("adapter", 1, lora_path) if lora_path else None
            )
            results.extend([o.outputs[0].text for o in outs])
        return results



MODEL_PATH = "/online1/sc100123/sc100123/Qwen3-4B-Ins/"
DATASET_PATH = "/online1/sc100123/sc100123/agentic_moderation/LLaMA-Factory/sd/data/train.parquet"

SYSTEM_PROMPT = """You are a careful, neutral annotator.

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

Output format (VERY IMPORTANT):
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


def gold_label_to_digit(label: Any) -> Optional[int]:
    """Map the gold label to a digit 0-4."""
    if isinstance(label, int):
        return label if 0 <= label <= 4 else None

    if isinstance(label, str):
        s = label.strip()
        if s in {"0", "1", "2", "3", "4"}:
            return int(s)

        mapping = {
            "s_against": 0,
            "against": 1,
            "stance_not_inferrable": 2,
            "stance_not_inferable": 2,
            "favor": 3,
            "s_favor": 4,
        }
        key = s.lower()
        if key in mapping:
            return mapping[key]

    return None


def build_base_input_from_row(row: dict) -> str:
    """
    Build the base input (without gold-label instructions) for a single example.

    If a 'user_prompt' column already exists, use it directly.
    Otherwise, construct a minimal structured prompt from basic fields.
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


def build_input_with_gold(row: dict) -> Optional[str]:
    """
    Build the full user input that includes the gold label and asks the model
    to produce a reasoning path leading to that label.
    """
    gold_digit = gold_label_to_digit(row.get("label", None))
    if gold_digit is None:
        return None

    base = build_base_input_from_row(row)

    extra_instruction = f"""

Now, based on the task definition above, the correct stance label for this specific example is the digit {gold_digit}.

Your job in this example is NOT to choose a label, but to explain why this specific label {gold_digit} is appropriate.

Follow these steps:
1. Inside a single pair of <reasoning>...</reasoning> tags, provide a clear, step-by-step explanation that justifies why label {gold_digit} is correct, referring to the TARGET_POST_TEXT, any provided context, and the label mapping.
2. Immediately after the closing </reasoning> tag, output exactly the single digit {gold_digit} again, with no extra text, no spaces, and no punctuation after it.
3. Do not output any other label than {gold_digit}.
"""

    return base + extra_instruction


def main(temperature: float, top_p: float) -> None:
    # Load data
    df = pd.read_parquet(DATASET_PATH)
    records = df.to_dict(orient="records")

    inputs = []
    meta = []  # keep aligned metadata for saving later

    for rec in records:
        user_input = build_input_with_gold(rec)
        if user_input is None:
            # skip rows with invalid or missing gold label
            continue
        inputs.append(user_input)
        meta.append(rec)

    if not inputs:
        print("No valid examples with gold labels found.")
        return

    # Initialize model
    llm = VLLM(model=MODEL_PATH)

    # Generate CoT with gold-consistent label
    outputs = llm.generate(
        inputs,
        system_prompt=SYSTEM_PROMPT,
        enable_thinking=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=1024,
    )

    # Build a lightweight training set: id, text, gold label (string + digit), and cot
    lite_records = []
    for rec, out in zip(meta, outputs):
        unit_id = rec.get("unit_id", None)
        target_text = rec.get("target_text", "")
        gold_label = rec.get("label", "")

        label_id = gold_label_to_digit(gold_label)

        if unit_id is not None:
            try:
                unit_id = int(unit_id)
            except Exception:
                unit_id = str(unit_id)

        lite_records.append(
            {
                "unit_id": unit_id,
                "target_text": str(target_text),
                "label": str(gold_label),
                "label_id": label_id,
                "cot": out,
            }
        )

    output_dir = "/online1/sc100123/sc100123/agentic_moderation/"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(
        output_dir,
        f"stance_train.json",
    )

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(lite_records, f, ensure_ascii=False, indent=2)

    print(f"Generated gold-consistent CoT training set saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling value.",
    )
    args = parser.parse_args()
    main(args.temperature, args.top_p)
