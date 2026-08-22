"""
DynaMath Evaluation Module

kcz358/DynaMath test split (5010 samples, 501 seeds × 10 variants)
- Mixed: float (~2960) + multiple_choice (~1740) + text (~310)
- Dynamic visual math reasoning benchmark
- MC scoring: Exact letter match (case-insensitive)
- Float scoring: Numeric comparison with tolerance (round to 4 decimal places)
- Text scoring: Normalized string comparison
- Reference: https://github.com/DynaMath/DynaMath
"""

import re
from datasets import load_dataset


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_items(cache_dir: str) -> list:
    """Load kcz358/DynaMath test split."""
    print("  Loading dynamath...")
    ds = load_dataset("kcz358/DynaMath", split="test", cache_dir=cache_dir)

    items = []
    for row in ds:
        img = row.get("decoded_image")
        if img is None:
            continue

        question = row.get("question", "")
        ground_truth = str(row.get("ground_truth", "")).strip()
        answer_type = row.get("answer_type", "float")

        # Determine if MC based on answer_type
        is_mc = (answer_type == "multiple choice")

        items.append({
            "image": img,
            "question": question,
            "answer": ground_truth,
            "is_mc": is_mc,
            "answer_type": answer_type,
        })

    mc_count = sum(1 for i in items if i["is_mc"])
    ff_count = len(items) - mc_count
    print(f"  Loaded {len(items)} samples (MC: {mc_count}, FreeForm: {ff_count})")
    return items


# ── Answer Extraction ──────────────────────────────────────────────────────────

def extract_answer(response: str, is_mc: bool = False) -> str:
    """
    Extract answer from model response.
    For MC: extract option letter (A-E).
    For Free-form (float/text): extract number or text.

    Priority:
    1. <answer>...</answer> tag
    2. Last \\boxed{...}
    3. MC letter / keyword patterns
    4. Last number in response
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

    # Get text after </think>
    if "</think>" in response:
        after_think = response.split("</think>")[-1].strip()
    else:
        after_think = response.strip()

    if not after_think:
        after_think = response.strip()

    # Step 2: Last \boxed{...} (for math responses)
    if not is_mc:
        boxed_answers = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', response)
        if boxed_answers:
            return boxed_answers[-1].strip()

    # Step 3: MC letter extraction
    if is_mc:
        letter = _extract_letter(after_think)
        if letter:
            return letter

        # Search full response for "answer is X" patterns
        all_matches = re.findall(r'(?:answer\s*(?:is|:)\s*)\(?([A-E])\)?', response, re.IGNORECASE)
        if all_matches:
            return all_matches[-1].upper()

        # Last resort: any standalone letter in after_think
        match = re.search(r'\b([A-E])\b', after_think)
        if match:
            return match.group(1).upper()

        return ""

    # Step 4: Free-form — keyword patterns
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

    # Step 5: Free-form — last line if short
    lines = [l.strip() for l in after_think.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) < 100:
            return last_line

    # Last number in response
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

def _normalize_text(s: str) -> str:
    """Normalize text answer for comparison."""
    s = str(s).strip().lower()
    # Remove punctuation except minus/dot
    s = re.sub(r"[^\w\s\.\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_correct(predicted: str, ground_truth: str, **kwargs) -> bool:
    """
    Check if prediction is correct.
    Dispatches based on answer_type:
    - "multiple choice": exact letter match
    - "float": numeric comparison (round to 4 decimal places)
    - "text": normalized string comparison
    """
    if not predicted:
        return False

    is_mc = kwargs.get("is_mc", False)
    answer_type = kwargs.get("answer_type", None)

    # Auto-detect if not specified
    if answer_type is None:
        if is_mc:
            answer_type = "multiple choice"
        else:
            # Try to parse as float
            try:
                float(ground_truth)
                answer_type = "float"
            except (ValueError, TypeError):
                answer_type = "text"

    if answer_type == "multiple choice" or is_mc:
        # MC: exact letter match
        pred = predicted.strip().upper()
        gt = ground_truth.strip().upper()

        if pred == gt:
            return True

        # Extract letter from pred
        pred_letter = re.match(r'^([A-E])\b', pred)
        gt_letter = re.match(r'^([A-E])\b', gt)

        if pred_letter and gt_letter:
            return pred_letter.group(1) == gt_letter.group(1)
        if pred_letter and len(gt) == 1:
            return pred_letter.group(1) == gt
        if gt_letter and len(pred) == 1:
            return pred == gt_letter.group(1)

        return False

    elif answer_type == "float":
        # Float: numeric comparison with tolerance
        try:
            # Extract numbers from prediction
            pred_nums = re.findall(r'-?\d+\.?\d*', predicted)
            if not pred_nums:
                return False
            pred_val = float(pred_nums[-1])
            gt_val = float(ground_truth)

            # Round to 4 decimal places for comparison
            if round(pred_val, 4) == round(gt_val, 4):
                return True

            # Relative tolerance (1%)
            if gt_val != 0 and abs(pred_val - gt_val) / abs(gt_val) < 0.01:
                return True

            # Absolute tolerance for very small numbers
            if abs(pred_val - gt_val) < 0.0001:
                return True

            return False
        except (ValueError, TypeError):
            return False

    else:
        # Text: normalized string comparison
        pred_norm = _normalize_text(predicted)
        gt_norm = _normalize_text(ground_truth)

        if not pred_norm or not gt_norm:
            return False

        # Exact match
        if pred_norm == gt_norm:
            return True

        # Containment (for short ground truth)
        if len(gt_norm) > 1 and gt_norm in pred_norm and len(pred_norm) < 50:
            return True

        # Try numeric comparison as fallback
        try:
            pred_nums = re.findall(r'-?\d+\.?\d*', pred_norm)
            gt_nums = re.findall(r'-?\d+\.?\d*', gt_norm)
            if pred_nums and gt_nums:
                if round(float(pred_nums[-1]), 4) == round(float(gt_nums[-1]), 4):
                    return True
        except (ValueError, TypeError):
            pass

        return False
