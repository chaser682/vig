"""
MathVision Evaluation Module
Reference: https://github.com/mathllm/MATH-V/tree/main/evaluation

MathLLMs/MathVision test split (3.04k samples)
- Mixed MC (has options) + Free-form (no options)
- Uses LaTeX-aware answer matching with symbolic equivalence
"""

import re
import os
from datasets import load_dataset


# ── Prompt Suffix ──────────────────────────────────────────────────────────────
# NOTE: No MC/Freeform suffix added — eval prompt must match training distribution.
# SYSTEM_PROMPT already instructs <think>+<answer> format.


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_items(cache_dir: str) -> list:
    """Load MathLLMs/MathVision test split."""
    print("  Loading mathvision...")
    ds = load_dataset("MathLLMs/MathVision", split="test", cache_dir=cache_dir)

    items = []
    for row in ds:
        # NOTE: "image" 字段是字符串路径（如 "images/1.jpg"，恒为真），真正的 PIL 图在
        # "decoded_image"。原先的 `row.get("image") or ...` 永远取到路径字符串，下游
        # resize_image 解析失败后回退为 224x224 白图 —— 2026-07-05 修复：优先 decoded_image。
        img = row.get("decoded_image") or None
        if img is None or isinstance(img, str):
            img = row.get("image")
        if isinstance(img, str):
            continue  # 无法解码的样本直接跳过，绝不回退白图
        if img is None:
            continue
        question = row.get("question", "")
        answer = str(row.get("answer", "")).strip()
        options = row.get("options")

        if options and isinstance(options, list) and len(options) > 1:
            valid = [o for o in options if o and str(o).strip()]
            if valid:
                opts_str = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(valid))
                question += "\n" + opts_str
                items.append({
                    "image": img,
                    "question": question,
                    "answer": answer,
                    "is_mc": True,
                    "options": valid,
                })
            else:
                items.append({
                    "image": img,
                    "question": question,
                    "answer": answer,
                    "is_mc": False,
                    "options": [],
                })
        else:
            items.append({
                "image": img,
                "question": question,
                "answer": answer,
                "is_mc": False,
                "options": [],
            })

    mc_count = sum(1 for i in items if i["is_mc"])
    ff_count = len(items) - mc_count
    print(f"  Loaded {len(items)} samples (MC: {mc_count}, FreeForm: {ff_count})")
    return items


# ── Answer Extraction (Reference: MATH-V evaluate.py) ─────────────────────────

def extract_answer(response: str, is_mc: bool = False) -> str:
    """
    Extract answer from model response.
    Priority:
    1. <answer>...</answer> tag
    2. Last \\boxed{...}
    3. MC letter detection
    4. "the answer is" keyword patterns
    5. Last number/expression
    """
    # Step 1: <answer> tag (our model's output format)
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        if answer_text:
            # For MC, try to extract just the letter
            if is_mc:
                letter = _extract_mc_letter(answer_text)
                if letter:
                    return letter
            return answer_text

    # Get text after </think> if present
    if "</think>" in response:
        after_think = response.split("</think>")[-1].strip()
    else:
        after_think = response.strip()

    # Step 2: Last \boxed{...} (common in math responses)
    boxed_answers = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', response)
    if boxed_answers:
        answer_text = boxed_answers[-1].strip()  # Take the LAST one (official behavior)
        if is_mc:
            letter = _extract_mc_letter(answer_text)
            if letter:
                return letter
        return answer_text

    # Step 3: MC letter detection (reference: MATH-V evaluate.py lines 24-27)
    if is_mc:
        letter = _extract_mc_letter(after_think)
        if letter:
            return letter

    # Step 4: "the answer is" keyword patterns
    flags = ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']
    for flag in flags:
        if flag in after_think.lower():
            parts = after_think.lower().split(flag)
            if len(parts) > 1:
                candidate = parts[-1].strip()
                # Clean up: take first line, first sentence
                candidate = candidate.split('\n')[0].split('. ')[0].strip()
                candidate = candidate.rstrip('.').strip()
                if candidate:
                    if is_mc:
                        letter = _extract_mc_letter(candidate)
                        if letter:
                            return letter
                    return candidate

    # Step 5: For MC, search more broadly
    if is_mc:
        # Search full response for letter patterns
        all_letters = re.findall(r'\b([A-E])\b', after_think)
        if all_letters:
            return all_letters[-1].upper()

    # Step 6: Last number in response
    numbers = re.findall(r'-?\d+\.?\d*', after_think)
    if numbers:
        return numbers[-1]

    return after_think.strip()[:100] if after_think else ""


def _extract_mc_letter(text: str) -> str:
    """Extract MC letter (A-E) from text."""
    text = text.strip()

    # Direct single letter
    clean = text.strip().strip('()').strip()
    if clean.upper() in "ABCDE" and len(clean) == 1:
        return clean.upper()

    # Pattern: ends with " A." or " (A)."
    for c in "ABCDE":
        if text.endswith(f" {c}.") or text.endswith(f" ({c})."):
            return c
        if text.startswith(f"{c}\n") or text.startswith(f"({c})\n"):
            return c
        if text.startswith(f"({c}) {c}\n"):
            return c

    # Pattern: (A) or just A in short text
    match = re.search(r'\(?([A-E])\)?', text)
    if match and len(text) < 20:
        return match.group(1).upper()

    # "answer is A" pattern
    match = re.search(r'(?:answer\s*(?:is|:)\s*)\(?([A-E])\)?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Standalone letter
    match = re.search(r'\b([A-E])\b', text)
    if match and len(text) < 30:
        return match.group(1).upper()

    return ""


# ── Answer Normalization (Reference: MATH-V utils.py _strip_string) ────────────

def _strip_string(s: str) -> str:
    """
    Normalize answer string for comparison.
    Reference: https://github.com/mathllm/MATH-V/blob/main/evaluation/utils.py
    """
    s = str(s)

    # Basic cleanup
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

    # Fix fracs: \frac12 -> \frac{1}{2}
    s = _fix_fracs(s)

    # Fix sqrt: \sqrt2 -> \sqrt{2}
    s = _fix_sqrt(s)

    # Handle equals/approx: take value after = or \approx
    if "=" in s:
        parts = s.split("=")
        s = parts[-1].strip()
    if "\\approx" in s:
        parts = s.split("\\approx")
        s = parts[-1].strip()

    # Remove leading/trailing whitespace and punctuation
    s = s.strip()
    s = s.rstrip('.')
    s = s.lstrip(':')
    s = s.strip()

    return s


def _fix_fracs(s: str) -> str:
    """Fix LaTeX fraction notation."""
    # \frac12 -> \frac{1}{2}
    while True:
        match = re.search(r'\\frac([^{])', s)
        if not match:
            break
        # Get position and fix
        pos = match.start()
        char = match.group(1)
        if char == ' ':
            s = s[:pos+5] + s[pos+6:]  # Remove space
        else:
            s = s[:pos+5] + '{' + char + '}' + s[pos+6:]

    # Fix second arg: \frac{1}2 -> \frac{1}{2}
    while True:
        match = re.search(r'\\frac\{[^}]*\}([^{])', s)
        if not match:
            break
        pos = match.start()
        end_brace = s.index('}', pos) + 1
        char = s[end_brace]
        if char == ' ':
            s = s[:end_brace] + s[end_brace+1:]
        else:
            s = s[:end_brace] + '{' + char + '}' + s[end_brace+1:]

    return s


def _fix_sqrt(s: str) -> str:
    """Fix LaTeX sqrt notation: \\sqrt2 -> \\sqrt{2}"""
    while True:
        match = re.search(r'\\sqrt([^{\\])', s)
        if not match:
            break
        pos = match.start()
        char = match.group(1)
        if char == ' ':
            s = s[:pos+5] + s[pos+6:]
        else:
            s = s[:pos+5] + '{' + char + '}' + s[pos+6:]
    return s


def _remove_right_units(s: str) -> str:
    """Remove units from the right side of the answer."""
    # Remove trailing text units
    s = re.sub(r'\s*(cm|mm|m|km|kg|g|lb|ft|in|degrees?|°)\s*$', '', s, flags=re.IGNORECASE)
    return s


# ── Answer Comparison (Reference: MATH-V utils.py is_equal) ────────────────────

def is_equal(pred: str, gt: str) -> bool:
    """
    Check if predicted answer equals ground truth.
    Reference: https://github.com/mathllm/MATH-V/blob/main/evaluation/utils.py

    Multi-stage comparison:
    1. Exact string match (after normalization)
    2. Numeric comparison (round to 2 decimals)
    3. Symbolic equivalence via latex2sympy (try/except)
    """
    if not pred or not gt:
        return False

    # Normalize both
    pred_norm = _strip_string(pred).lower().strip()
    gt_norm = _strip_string(gt).lower().strip()

    # Remove any remaining answer tags
    pred_norm = re.sub(r'</?answer>', '', pred_norm).strip()
    gt_norm = re.sub(r'</?answer>', '', gt_norm).strip()

    if not pred_norm or not gt_norm:
        return False

    # Stage 1: Exact string match
    if pred_norm == gt_norm:
        return True

    # Also try without spaces
    if pred_norm.replace(' ', '') == gt_norm.replace(' ', ''):
        return True

    # Stage 2: Numeric comparison (round to 2 decimal places, only if pred is short)
    if len(pred_norm) < 50:
        try:
            pred_nums = re.findall(r'-?\d+\.?\d*', pred_norm)
            gt_nums = re.findall(r'-?\d+\.?\d*', gt_norm)
            if pred_nums and gt_nums:
                pred_val = float(pred_nums[-1])
                gt_val = float(gt_nums[-1])
                if round(pred_val, 2) == round(gt_val, 2):
                    return True
                # Relative tolerance
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

    # Stage 4: Containment check (less strict)
    if gt_norm in pred_norm and len(gt_norm) > 1:
        return True

    return False


def check_correct(predicted: str, ground_truth: str, is_mc: bool = False, options: list = None) -> bool:
    """
    Check if prediction is correct.
    For MC: check letter match AND option value match.
    For free-form: use is_equal with full normalization.
    """
    if not predicted:
        return False

    if is_mc:
        # Check letter match
        pred_letter = predicted.strip().upper()
        gt_upper = ground_truth.strip().upper()

        # Direct letter match
        if len(pred_letter) == 1 and pred_letter in "ABCDE":
            # GT might be letter or might be full text
            gt_letter = re.match(r'^([A-E])$', gt_upper)
            if gt_letter and pred_letter == gt_letter.group(1):
                return True
            # GT might start with letter
            gt_letter = re.match(r'^([A-E])\b', gt_upper)
            if gt_letter and pred_letter == gt_letter.group(1):
                return True

        # If we have options, check if predicted value matches GT option value
        if options:
            # pred is a letter -> get option value
            if len(pred_letter) == 1 and pred_letter in "ABCDE":
                idx = ord(pred_letter) - ord('A')
                if idx < len(options):
                    pred_value = str(options[idx]).strip()
                    if is_equal(pred_value, ground_truth):
                        return True

            # GT is a letter -> compare with pred letter
            if len(gt_upper) == 1 and gt_upper in "ABCDE":
                if pred_letter == gt_upper:
                    return True
                # Also check if pred value matches the GT option value
                gt_idx = ord(gt_upper) - ord('A')
                if gt_idx < len(options):
                    gt_value = str(options[gt_idx]).strip()
                    if is_equal(predicted, gt_value):
                        return True

        # Fallback: direct comparison
        if is_equal(predicted, ground_truth):
            return True

        return False
    else:
        # Free-form: use full is_equal
        return is_equal(predicted, ground_truth)
