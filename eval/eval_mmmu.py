"""
MMMU Evaluation Module

MMMU/MMMU validation split (900 samples, 30 subject configs)
- Mixed: 847 MC (multiple-choice) + 53 Open (free-form)
- Multimodal understanding across 30 academic subjects
- MC scoring: Exact letter match (case-insensitive)
- Open scoring: Normalized string / numeric comparison
"""

import re
import ast
from datasets import load_dataset, concatenate_datasets


# ── Prompt Suffix ──────────────────────────────────────────────────────────────
# NOTE: No MC_PROMPT_SUFFIX added — eval prompt must match training distribution.
# SYSTEM_PROMPT already instructs <think>+<answer> format.


# ── Dataset Loading ────────────────────────────────────────────────────────────

MMMU_CONFIGS = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
    'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry',
    'Clinical_Medicine', 'Computer_Science', 'Design',
    'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
    'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]


def load_items(cache_dir: str) -> list:
    """Load MMMU/MMMU validation split (30 subject configs concatenated)."""
    print("  Loading mmmu...")
    all_ds = []
    for cfg in MMMU_CONFIGS:
        try:
            d = load_dataset("MMMU/MMMU", cfg, split="validation", cache_dir=cache_dir)
            all_ds.append(d)
        except Exception:
            pass

    if not all_ds:
        print("  ERROR: No MMMU configs loaded!")
        return []

    ds = concatenate_datasets(all_ds)

    items = []
    for row in ds:
        # Get first available image (MMMU supports up to 7 images)
        img = None
        for key in ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]:
            img = row.get(key)
            if img is not None:
                break
        if img is None:
            continue

        question = row.get("question", "")
        answer = str(row.get("answer", "")).strip()
        options_str = row.get("options", "")
        question_type = row.get("question_type", "multiple-choice")

        # Determine if MC or open based on question_type field
        is_mc = (question_type == "multiple-choice")

        # Parse options and append to question (only for MC)
        if is_mc and options_str:
            try:
                choices = ast.literal_eval(options_str)
                if isinstance(choices, (list, tuple)) and choices:
                    question += "\n" + "\n".join(
                        f"({chr(65+i)}) {c}" for i, c in enumerate(choices)
                    )
            except Exception:
                pass

        items.append({
            "image": img,
            "question": question,
            "answer": answer,
            "is_mc": is_mc,
        })

    mc_count = sum(1 for i in items if i["is_mc"])
    ff_count = len(items) - mc_count
    print(f"  Loaded {len(items)} samples (MC: {mc_count}, FreeForm: {ff_count})")
    return items


# ── Answer Extraction ──────────────────────────────────────────────────────────

def extract_answer(response: str, is_mc: bool = True) -> str:
    """
    Extract answer from model response.
    For MC: extract option letter (A-E).
    For Open: extract the answer text/number.

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

    # Step 5: Free-form — last line
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

def _normalize_open_answer(s: str) -> str:
    """Normalize open-ended answer for comparison."""
    s = str(s).strip()
    s = s.lower()
    # Remove common LaTeX noise
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\text{", "").replace("}", "")
    s = s.replace("\\", "")
    # Remove punctuation except minus/dot for numbers
    s = re.sub(r"[^\w\s\.\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_equal_open(pred: str, gt: str) -> bool:
    """Check if open-ended answers are equivalent."""
    if not pred or not gt:
        return False

    pred_norm = _normalize_open_answer(pred)
    gt_norm = _normalize_open_answer(gt)

    if not pred_norm or not gt_norm:
        return False

    # Exact string match
    if pred_norm == gt_norm:
        return True

    # Numeric comparison (only if pred is short)
    if len(pred_norm) < 50:
        try:
            pred_nums = re.findall(r'-?\d+\.?\d*', pred_norm)
            gt_nums = re.findall(r'-?\d+\.?\d*', gt_norm)
            if pred_nums and gt_nums:
                pred_val = float(pred_nums[-1])
                gt_val = float(gt_nums[-1])
                if round(pred_val, 2) == round(gt_val, 2):
                    return True
                if gt_val != 0 and abs(pred_val - gt_val) / abs(gt_val) < 0.01:
                    return True
        except (ValueError, TypeError):
            pass

    # Containment (gt in pred, for short gt)
    if len(gt_norm) > 1 and gt_norm in pred_norm and len(pred_norm) < 50:
        return True

    return False


def check_correct(predicted: str, ground_truth: str, **kwargs) -> bool:
    """
    Check if prediction is correct.
    Supports both MC and open-ended questions.

    For MC: case-insensitive letter comparison.
    For Open: normalized string / numeric comparison.
    """
    if not predicted:
        return False

    is_mc = kwargs.get("is_mc", None)

    # Auto-detect if not specified
    if is_mc is None:
        gt = ground_truth.strip()
        is_mc = (len(gt) == 1 and gt.upper() in "ABCDE")

    if is_mc:
        pred = predicted.strip().upper()
        gt = ground_truth.strip().upper()

        # Direct letter match
        if pred == gt:
            return True

        # GT might have extra content
        gt_letter = re.match(r'^([A-E])\b', gt)
        if gt_letter and pred == gt_letter.group(1):
            return True

        # Pred might have extra content
        pred_letter = re.match(r'^([A-E])\b', pred)
        if pred_letter and pred_letter.group(1) == gt:
            return True

        return False
    else:
        # Open-ended: use normalized comparison
        return _is_equal_open(predicted, ground_truth)
