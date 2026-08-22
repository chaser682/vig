"""
OCRBench Evaluation Module

echo840/OCRBench test split (1000 samples)
- Document / scene text understanding (non-math, rebuttal benchmark):
  text recognition, scene-text VQA, doc VQA, KIE, handwritten math expression
- All short-answer (free-form); each question may have multiple acceptable answers
- Scoring (follows official OCRBench protocol): correct if any ground-truth
  string is contained in the prediction (case-insensitive; spaces stripped
  for HME100k handwritten-expression samples)
"""

import re
from datasets import load_dataset


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_items(cache_dir: str) -> list:
    """Load echo840/OCRBench test split."""
    print("  Loading ocrbench...")
    ds = load_dataset("echo840/OCRBench", split="test", cache_dir=cache_dir)

    items = []
    for row in ds:
        img = row.get("image")
        if img is None:
            continue
        question = str(row.get("question", "")).strip()
        answer = row.get("answer", row.get("answers", ""))
        if not question or answer is None:
            continue

        # Normalize answer to a list of acceptable strings
        if isinstance(answer, str):
            answers = [answer]
        elif isinstance(answer, (list, tuple)):
            answers = [str(a) for a in answer if str(a).strip()]
        else:
            answers = [str(answer)]
        if not answers:
            continue

        items.append({
            "image": img,
            "question": question,
            "answer": answers,  # list of acceptable ground-truth strings
            "is_mc": False,
            "dataset_name": str(row.get("dataset", "")),
            "question_type": str(row.get("question_type", "")),
        })

    print(f"  Loaded {len(items)} samples (MC: 0, FreeForm: {len(items)})")
    return items


# ── Answer Extraction ──────────────────────────────────────────────────────────

def extract_answer(response: str, is_mc: bool = False) -> str:
    """
    Extract short answer text from model response.

    Priority:
    1. <answer>...</answer> tag
    2. Text after </think> (full tail — scoring is containment-based,
       so keeping more context is safe)
    """
    # Step 1: <answer> tag
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        if answer_text:
            return answer_text

    # Step 2: text after </think>
    if "</think>" in response:
        after_think = response.split("</think>")[-1].strip()
    else:
        after_think = response.strip()

    if not after_think:
        after_think = response.strip()

    # Strip stray tags
    after_think = re.sub(r'</?answer>', '', after_think).strip()

    # Keyword patterns
    flags = ['the final answer is', 'the answer is', 'the correct answer is']
    for flag in flags:
        idx = after_think.lower().rfind(flag)
        if idx != -1:
            candidate = after_think[idx + len(flag):].strip()
            candidate = candidate.split('\n')[0].strip().rstrip('.').strip()
            if candidate:
                return candidate

    # Containment scoring: return the tail (capped) so short GT can match
    return after_think[:500]


# ── Answer Checking ────────────────────────────────────────────────────────────

def _normalize(s: str, strip_spaces: bool = False) -> str:
    s = str(s).strip().lower().replace("\n", " ")
    if strip_spaces:
        s = s.replace(" ", "").replace("$", "")
    else:
        s = re.sub(r"\s+", " ", s)
    return s


def check_correct(predicted, ground_truth, **kwargs) -> bool:
    """
    OCRBench official-style scoring: correct if ANY acceptable ground-truth
    string is a substring of the prediction (case-insensitive).
    For handwritten math expressions (HME100k), spaces are stripped first.

    ground_truth: list of acceptable strings (or a single string).
    """
    if not predicted:
        return False

    if isinstance(ground_truth, str):
        gts = [ground_truth]
    elif isinstance(ground_truth, (list, tuple)):
        gts = [str(g) for g in ground_truth]
    else:
        gts = [str(ground_truth)]

    dataset_name = kwargs.get("dataset_name", "")
    question_type = kwargs.get("question_type", "")
    strip_spaces = (
        "HME100k" in dataset_name or "HME100K" in dataset_name
        or "Handwritten Mathematical Expression" in question_type
    )

    pred_norm = _normalize(predicted, strip_spaces=strip_spaces)
    if not pred_norm:
        return False

    for gt in gts:
        gt_norm = _normalize(gt, strip_spaces=strip_spaces)
        if gt_norm and gt_norm in pred_norm:
            return True

    return False
