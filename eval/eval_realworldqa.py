"""
RealWorldQA Evaluation Module

xai-org/RealworldQA test split (765 samples)
- Real-world spatial / scene understanding (non-math, rebuttal benchmark)
- Mostly Multiple Choice (options embedded in question text) + a few short answers
- MC scoring: exact letter match (case-insensitive)
- Short-answer scoring: normalized string / numeric comparison
"""

import re
from datasets import load_dataset


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_items(cache_dir: str) -> list:
    """Load xai-org/RealworldQA test split."""
    print("  Loading realworldqa...")
    ds = load_dataset("xai-org/RealworldQA", split="test", cache_dir=cache_dir)

    items = []
    for row in ds:
        img = row.get("image")
        if img is None:
            continue
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue

        # MC iff ground truth is a single option letter (options are embedded
        # in the question text by the dataset itself).
        is_mc = len(answer) == 1 and answer.upper() in "ABCDE"

        items.append({
            "image": img,
            "question": question,
            "answer": answer,
            "is_mc": is_mc,
        })

    mc_count = sum(1 for i in items if i["is_mc"])
    print(f"  Loaded {len(items)} samples (MC: {mc_count}, FreeForm: {len(items) - mc_count})")
    return items


# ── Answer Extraction ──────────────────────────────────────────────────────────

def extract_answer(response: str, is_mc: bool = True) -> str:
    """
    Extract answer from model response.
    For MC: extract option letter (A-E).
    For short answer: extract the answer text/number.

    Priority:
    1. <answer>...</answer> tag
    2. Text after </think>
    3. MC letter / keyword / number patterns
    """
    # Step 1: <answer> tag
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        if answer_text:
            if is_mc:
                letter = _extract_letter(answer_text)
                if letter:
                    return letter
            return answer_text

    # Step 2: Get text after </think>
    if "</think>" in response:
        after_think = response.split("</think>")[-1].strip()
    else:
        after_think = response.strip()

    if not after_think:
        after_think = response.strip()

    # Step 3: MC letter extraction
    if is_mc:
        letter = _extract_letter(after_think)
        if letter:
            return letter

        # Search full response for "answer is X" patterns
        all_matches = re.findall(r'(?:answer\s*(?:is|:)\s*)\(?([A-E])\)?', response, re.IGNORECASE)
        if all_matches:
            return all_matches[-1].upper()

        # Last resort: any letter in after_think
        match = re.search(r'([A-E])', after_think)
        if match:
            return match.group(1).upper()

        return ""

    # Step 4: Short answer — keyword patterns
    flags = ['the final answer is', 'the answer is', 'the correct answer is']
    for flag in flags:
        if flag in after_think.lower():
            parts = after_think.lower().split(flag)
            if len(parts) > 1:
                candidate = parts[-1].strip()
                candidate = candidate.split('\n')[0].split('. ')[0].strip()
                candidate = candidate.rstrip('.').strip()
                if candidate:
                    return candidate

    # Step 5: Short answer — last line
    lines = [l.strip() for l in after_think.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) < 100:
            return last_line

    # Last number
    numbers = re.findall(r'-?\d+\.?\d*', after_think)
    if numbers:
        return numbers[-1]

    return after_think.strip()[:100] if after_think else ""


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

def _normalize_short_answer(s: str) -> str:
    """Normalize short answer for comparison."""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s\.\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_correct(predicted: str, ground_truth: str, **kwargs) -> bool:
    """
    Check if prediction is correct.
    MC: case-insensitive letter comparison.
    Short answer: normalized string / numeric comparison.
    """
    if not predicted:
        return False

    is_mc = kwargs.get("is_mc", None)
    if is_mc is None:
        gt = str(ground_truth).strip()
        is_mc = (len(gt) == 1 and gt.upper() in "ABCDE")

    if is_mc:
        pred = predicted.strip().upper()
        gt = str(ground_truth).strip().upper()

        if pred == gt:
            return True

        gt_letter = re.match(r'^([A-E])\b', gt)
        if gt_letter and pred == gt_letter.group(1):
            return True

        pred_letter = re.match(r'^([A-E])\b', pred)
        if pred_letter and pred_letter.group(1) == gt:
            return True

        return False

    # Short answer: normalized exact / numeric / containment match
    pred_norm = _normalize_short_answer(predicted)
    gt_norm = _normalize_short_answer(ground_truth)
    if not pred_norm or not gt_norm:
        return False

    if pred_norm == gt_norm:
        return True

    # Numeric comparison
    try:
        pred_nums = re.findall(r'-?\d+\.?\d*', pred_norm)
        gt_nums = re.findall(r'-?\d+\.?\d*', gt_norm)
        if pred_nums and gt_nums:
            if round(float(pred_nums[-1]), 2) == round(float(gt_nums[-1]), 2):
                return True
    except (ValueError, TypeError):
        pass

    # Containment (short gt in short pred)
    if len(gt_norm) > 1 and gt_norm in pred_norm and len(pred_norm) < 50:
        return True

    return False
