"""
Geo3K Evaluation Module

SunnyLin/geo3k-tikz test split (601 samples)
- All free-form geometry problems (numerical / mathematical expression answers)
- Geometry problem solving with diagrams
- Scoring: Numeric + LaTeX-aware equivalence matching
- Reuses eval_mathvision.is_equal() for answer comparison
- Reference: Geometry3K benchmark
"""

import re
from datasets import load_dataset

# Import is_equal from eval_mathvision for LaTeX-aware answer comparison
import eval_mathvision


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_items(cache_dir: str) -> list:
    """Load SunnyLin/geo3k-tikz test split."""
    print("  Loading geo3k...")
    ds = load_dataset("SunnyLin/geo3k-tikz", split="test", cache_dir=cache_dir)

    items = []
    for row in ds:
        img = row.get("image")
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
            "is_mc": False,
        })

    print(f"  Loaded {len(items)} samples (MC: 0, FreeForm: {len(items)})")
    return items


# ── Answer Extraction ──────────────────────────────────────────────────────────

def extract_answer(response: str, is_mc: bool = False) -> str:
    """
    Extract free-form math answer from model response.
    Priority:
    1. <answer>...</answer> tag
    2. Last \\boxed{...}
    3. "the answer is" keyword patterns
    4. Last number/expression
    """
    # Step 1: <answer> tag
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        if answer_text:
            return answer_text

    # Get text after </think>
    if "</think>" in response:
        after_think = response.split("</think>")[-1].strip()
    else:
        after_think = response.strip()

    # Step 2: Last \boxed{...}
    boxed_answers = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', response)
    if boxed_answers:
        return boxed_answers[-1].strip()

    # Step 3: Keyword patterns
    flags = ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']
    for flag in flags:
        if flag in after_think.lower():
            parts = after_think.lower().split(flag)
            if len(parts) > 1:
                candidate = parts[-1].strip()
                candidate = candidate.split('\n')[0].split('. ')[0].strip()
                candidate = candidate.rstrip('.').strip()
                if candidate:
                    return candidate

    # Step 4: Last line if short
    lines = [l.strip() for l in after_think.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) < 100:
            # Try to extract just the number/expression from it
            numbers = re.findall(r'-?\d+\.?\d*', last_line)
            if numbers and len(last_line) < 30:
                return last_line
            elif numbers:
                return numbers[-1]

    # Step 5: Last number in response
    numbers = re.findall(r'-?\d+\.?\d*', after_think)
    if numbers:
        return numbers[-1]

    return after_think.strip()[:100] if after_think else ""


# ── Answer Checking ────────────────────────────────────────────────────────────

def check_correct(predicted: str, ground_truth: str, **kwargs) -> bool:
    """
    Check if free-form geometry answer is correct.
    Uses eval_mathvision.is_equal() for LaTeX-aware equivalence.
    """
    if not predicted:
        return False

    return eval_mathvision.is_equal(predicted, ground_truth)
