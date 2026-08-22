"""
We-Math Evaluation Module

We-Math/We-Math testmini (1.74k samples)
- All Multiple Choice questions
- Visual math reasoning
- Scoring: Exact letter match (case-insensitive)
"""

import re
from datasets import load_dataset


# ── Prompt Suffix ──────────────────────────────────────────────────────────────
# NOTE: No MC_PROMPT_SUFFIX added — eval prompt must match training distribution.
# SYSTEM_PROMPT already instructs <think>+<answer> format.


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_items(cache_dir: str) -> list:
    """Load We-Math/We-Math testmini split."""
    print("  Loading wemath...")
    ds = load_dataset("We-Math/We-Math", split="testmini", cache_dir=cache_dir)

    items = []
    for row in ds:
        img = row.get("image_path")
        if img is None:
            continue
        question = row.get("question", "")
        option_str = row.get("option", "")
        answer = str(row.get("answer", "")).strip()

        if option_str:
            question += "\n" + option_str

        items.append({
            "image": img,
            "question": question,
            "answer": answer,
            "is_mc": True,
        })

    print(f"  Loaded {len(items)} samples (MC: {len(items)}, FreeForm: 0)")
    return items


# ── Answer Extraction ──────────────────────────────────────────────────────────

def extract_answer(response: str, is_mc: bool = True) -> str:
    """
    Extract MC answer letter from model response.
    Priority:
    1. <answer>...</answer> tag
    2. Text after </think>
    3. MC letter pattern matching
    """
    # Step 1: <answer> tag
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        letter = _extract_letter(answer_text)
        if letter:
            return letter

    # Step 2: Get text after </think>
    if "</think>" in response:
        after_think = response.split("</think>")[-1].strip()
    else:
        after_think = response.strip()

    if not after_think:
        after_think = response.strip()

    # Step 3: Extract letter from answer part
    letter = _extract_letter(after_think)
    if letter:
        return letter

    # Step 4: Search full response for "answer is X" patterns
    all_matches = re.findall(r'(?:answer\s*(?:is|:)\s*)\(?([A-E])\)?', response, re.IGNORECASE)
    if all_matches:
        return all_matches[-1].upper()

    # Last resort: any letter in after_think
    match = re.search(r'([A-E])', after_think)
    if match:
        return match.group(1).upper()

    return ""


def _extract_letter(text: str) -> str:
    """Extract a single MC letter (A-E) from text."""
    text = text.strip()

    # Direct single letter
    clean = text.strip().strip('()').strip()
    if clean.upper() in "ABCDE" and len(clean) == 1:
        return clean.upper()

    # Short text with (A) or A pattern
    match = re.search(r'\(?([A-E])\)?', text)
    if match and len(text) < 20:
        return match.group(1).upper()

    # "The answer is A" / "Answer: B"
    match = re.search(r'(?:answer\s*(?:is|:)\s*)\(?([A-E])\)?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Letter at end of text
    match = re.search(r'\b([A-E])\b[.\s]*$', text)
    if match:
        return match.group(1).upper()

    # First standalone letter in short text
    if len(text) < 50:
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1).upper()

    return ""


# ── Answer Checking ────────────────────────────────────────────────────────────

def check_correct(predicted: str, ground_truth: str, **kwargs) -> bool:
    """
    Check if MC answer is correct.
    Case-insensitive letter comparison.
    """
    if not predicted:
        return False

    pred = predicted.strip().upper()
    gt = ground_truth.strip().upper()

    # Direct letter match
    if pred == gt:
        return True

    # GT might have extra content (e.g., "A. answer text")
    gt_letter = re.match(r'^([A-E])\b', gt)
    if gt_letter and pred == gt_letter.group(1):
        return True

    # Pred might have extra content
    pred_letter = re.match(r'^([A-E])\b', pred)
    if pred_letter and pred_letter.group(1) == gt:
        return True

    return False
