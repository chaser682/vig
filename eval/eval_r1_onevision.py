"""
R1-Onevision Evaluation Module

Fancy-MLLM/R1-Onevision-Bench (942 samples)
- Mixed: 783 MC (has choices field) + 159 Free-form (no choices, math answers)
- MC scoring: Exact letter match (case-insensitive)
- Free-form scoring: LaTeX-aware matching (numeric + symbolic equivalence)
- Note: Images are stored as base64 encoded strings
"""

import re
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset


# ── Prompt Suffix ──────────────────────────────────────────────────────────────
# NOTE: No MC_PROMPT_SUFFIX added — eval prompt must match training distribution.
# SYSTEM_PROMPT already instructs <think>+<answer> format.


# ── Dataset Loading ────────────────────────────────────────────────────────────

def _decode_base64_image(b64_str):
    """Decode base64 string to PIL Image."""
    try:
        img_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(img_data)).convert("RGB")
    except Exception:
        return None


def load_items(cache_dir: str) -> list:
    """Load Fancy-MLLM/R1-Onevision-Bench train split (mixed MC + Free-form)."""
    print("  Loading r1_onevision...")
    ds = load_dataset("Fancy-MLLM/R1-Onevision-Bench", split="train", cache_dir=cache_dir)

    items = []
    for row in ds:
        img_b64 = row.get("image")
        if not img_b64:
            continue

        question = row.get("question", "")
        choices = row.get("choices", None)
        answer = str(row.get("answer", "")).strip()

        # Determine if MC or free-form based on choices field
        is_mc = bool(choices and str(choices).strip())

        if is_mc and choices not in question:
            question += "\n" + str(choices)

        image = _decode_base64_image(img_b64)
        if image is None:
            continue

        items.append({
            "image": image,
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
    For Free-form: extract math expression / number.

    Priority:
    1. <answer>...</answer> tag
    2. Last \\boxed{...} (common in math responses)
    3. Text after </think>
    4. Pattern matching (MC letter / keyword / number)
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

    # Step 2: Last \boxed{...} (for math free-form)
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

    # Step 5: Free-form — last number/expression
    # Try to get the last line as answer
    lines = [l.strip() for l in after_think.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # If last line is short, use it directly
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


# ── Answer Normalization (for free-form) ──────────────────────────────────────

def _strip_string(s: str) -> str:
    """Normalize answer string for comparison (LaTeX-aware)."""
    s = str(s)
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac")
    s = s.replace("dfrac", "frac")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("^{\\circ}", "")
    s = s.replace("^\\circ", "")
    s = s.replace("\\$", "")
    s = s.replace("$", "")
    s = s.replace("\\%", "")
    s = s.replace("%", "")
    s = s.replace(" .", " 0.")
    s = s.replace("{.", "{0.")

    # Remove \text{...} units
    s = re.sub(r'\\text\{[^}]*\}', '', s)
    # Remove common units
    for unit in ['{km}', '{m}', '{cm}', '{mm}', '{m^3}', '{cm^2}', '{units}',
                 '{kg}', '{g}', '{lb}', '{ft}', '{in}', '{degrees}', '{degree}']:
        s = s.replace(unit, '')

    # Remove trailing units
    s = re.sub(r'\s*(cm|mm|m|km|kg|g|lb|ft|in|degrees?|°|L)\s*$', '', s, flags=re.IGNORECASE)

    # Handle equals/approx
    if "=" in s:
        s = s.split("=")[-1].strip()
    if "\\approx" in s:
        s = s.split("\\approx")[-1].strip()

    s = s.strip().rstrip('.').lstrip(':').strip()
    return s


# ── Answer Checking ────────────────────────────────────────────────────────────

def _is_equal_freeform(pred: str, gt: str) -> bool:
    """
    Check if free-form answers are equivalent.
    Multi-stage: string match → numeric → latex2sympy.
    """
    if not pred or not gt:
        return False

    pred_norm = _strip_string(pred).lower().strip()
    gt_norm = _strip_string(gt).lower().strip()

    if not pred_norm or not gt_norm:
        return False

    # Stage 1: Exact string match
    if pred_norm == gt_norm:
        return True
    if pred_norm.replace(' ', '') == gt_norm.replace(' ', ''):
        return True

    # Stage 2: Numeric comparison (only if pred is short enough to be an answer)
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
        except (ValueError, TypeError, IndexError):
            pass

    # Stage 3: Symbolic equivalence via latex2sympy (only if pred looks like math)
    if len(pred_norm) < 80:
        try:
            from latex2sympy2 import latex2sympy
            pred_expr = eval(str(latex2sympy(pred_norm)))
            gt_expr = eval(str(latex2sympy(gt_norm)))
            if round(float(pred_expr), 2) == round(float(gt_expr), 2):
                return True
        except Exception:
            pass

    # Stage 4: Containment
    if gt_norm in pred_norm and len(gt_norm) > 1:
        return True

    return False


def check_correct(predicted: str, ground_truth: str, **kwargs) -> bool:
    """
    Check if prediction is correct.
    Automatically detects MC vs free-form based on ground_truth format.

    For MC (ground_truth is single letter A-E): letter comparison.
    For Free-form: LaTeX-aware equivalence checking.
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
        # Free-form: use LaTeX-aware equivalence
        return _is_equal_freeform(predicted, ground_truth)
