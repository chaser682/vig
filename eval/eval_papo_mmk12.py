"""
PAPO MMK12 Evaluation Module

PAPOGalaxy/PAPO_MMK12_test (4000 samples)
- All Multiple Choice questions (A/B/C/D)
- K12-level mathematics and geometry problems with visual context
- Scoring: Exact letter match (case-insensitive)
- Reference: https://huggingface.co/datasets/PAPOGalaxy/PAPO_MMK12_test
"""

import re
from datasets import load_dataset


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_items(cache_dir: str) -> list:
    """Load PAPOGalaxy/PAPO_MMK12_test train split (only available split)."""
    print("  Loading papo_mmk12...")
    ds = load_dataset("PAPOGalaxy/PAPO_MMK12_test", split="train", cache_dir=cache_dir)

    items = []
    for row in ds:
        images = row.get("images")
        if not images or len(images) == 0:
            continue

        img = images[0]  # Take the first image from the list
        if img is None:
            continue

        problem = row.get("problem", "")
        answer = str(row.get("answer", "")).strip()

        # Remove <image> tag from problem (vLLM handles image separately)
        question = problem.replace("<image>", "").strip()

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
    all_matches = re.findall(r'(?:answer\s*(?:is|:)\s*)\(?([A-D])\)?', response, re.IGNORECASE)
    if all_matches:
        return all_matches[-1].upper()

    # Last resort: any letter in after_think
    match = re.search(r'([A-D])', after_think)
    if match:
        return match.group(1).upper()

    return ""


def _extract_letter(text: str) -> str:
    """Extract a single MC letter (A-D) from text."""
    text = text.strip()

    # Direct single letter
    clean = text.strip().strip('()').strip()
    if clean.upper() in "ABCD" and len(clean) == 1:
        return clean.upper()

    # Short text with (A) or A pattern
    match = re.search(r'\(?([A-D])\)?', text)
    if match and len(text) < 20:
        return match.group(1).upper()

    # "The answer is A" / "Answer: B"
    match = re.search(r'(?:answer\s*(?:is|:)\s*)\(?([A-D])\)?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Letter at end of text
    match = re.search(r'\b([A-D])\b[.\s]*$', text)
    if match:
        return match.group(1).upper()

    # First standalone letter in short text
    if len(text) < 50:
        match = re.search(r'\b([A-D])\b', text)
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
    gt_letter = re.match(r'^([A-D])\b', gt)
    if gt_letter and pred == gt_letter.group(1):
        return True

    # Pred might have extra content
    pred_letter = re.match(r'^([A-D])\b', pred)
    if pred_letter and pred_letter.group(1) == gt:
        return True

    return False
