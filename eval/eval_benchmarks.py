"""
VIG evaluation across the benchmarks reported in the paper (unified dispatcher).
Each dataset has its own module with specialized answer extraction and checking.

Main benchmarks (Table 1):
- We-Math/We-Math testmini (1,740): MC, visual math reasoning
- MMMU/MMMU validation (900): MC, multi-discipline multimodal understanding
- MathLLMs/MathVision test (3,040): mixed MC + free-form (MATH-V official eval)
- kcz358/DynaMath test (5,010): mixed MC + float + text, robustness under perturbation
- PAPOGalaxy/PAPO_MMK12_test (2,000): MC, K-12 multi-subject mathematics
- SunnyLin/geo3k-tikz test (601): free-form, geometry

Subject-level evaluation (Table 2):
- Fancy-MLLM/R1-Onevision-Bench (942): physics / math / biology / chemistry / deduction

Non-mathematical visual tasks (Table 3):
- xai-org/RealworldQA (765): real-world scene understanding
- echo840/OCRBench (1,000): text recognition and document understanding

Scoring modes (--judge_mode):
- "rule" (default): rule-based matching (exact letter / numeric tolerance / LaTeX).
  This is the protocol reported in the paper and requires no API access.
- "api" / "both": optional LLM-as-judge; reads credentials from the environment
  (see eval/eval_api_judge.py).

Usage:
    python eval/eval_benchmarks.py --model_path <checkpoint> --output_dir output/eval_vig
"""

import os
import sys
import json
import re
import argparse
import base64
from pathlib import Path
from io import BytesIO

from PIL import Image
from vllm import LLM, SamplingParams

# Add eval directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_wemath
import eval_r1_onevision
import eval_mmmu
import eval_mathvision
import eval_dynamath
import eval_papo_mmk12
import eval_geo3k
import eval_realworldqa
import eval_ocrbench
from eval_api_judge import ApiJudge


# ── Dataset Registry ───────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "wemath": eval_wemath,
    "r1_onevision": eval_r1_onevision,
    "mmmu": eval_mmmu,
    "mathvision": eval_mathvision,
    "dynamath": eval_dynamath,
    "papo_mmk12": eval_papo_mmk12,
    "geo3k": eval_geo3k,
    # Non-mathematical visual tasks (scene understanding / document & text OCR)
    "realworldqa": eval_realworldqa,
    "ocrbench": eval_ocrbench,
}


# ── Shared Utilities ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. "
    "Think step by step, focusing on information directly relevant to solving the problem. "
    "Avoid redundant or generic statements. "
    "Format your response as:\n"
    "<think>\nYour concise step-by-step reasoning\n</think>\n"
    "<answer>\nYour final answer\n</answer>"
)

# A5 prompt baseline：训练-free 的"视觉接地+简洁"系统提示，用于展示 prompt 无法替代 RL 信号
SYSTEM_PROMPT_CONCISE_VISUAL = (
    "You are a helpful visual reasoning assistant. "
    "Keep your reasoning as SHORT as possible: only include steps where the image provides "
    "critical information, read the needed values from the image directly, and skip all "
    "meta-commentary, restatements of the question, and self-verification. "
    "Format your response as:\n"
    "<think>\nYour concise step-by-step reasoning\n</think>\n"
    "<answer>\nYour final answer\n</answer>"
)

SYSTEM_PROMPT_MAP = {
    "default": SYSTEM_PROMPT,
    "concise_visual": SYSTEM_PROMPT_CONCISE_VISUAL,
}


def decode_base64_image(b64_str):
    """Decode base64 string to PIL Image."""
    try:
        img_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(img_data)).convert("RGB")
    except Exception:
        return None


def resize_image(image, max_pixels=768*768):
    """Resize image to limit token count in vLLM."""
    if isinstance(image, str):
        if len(image) > 1000:
            image = decode_base64_image(image)
            if image is None:
                return Image.new('RGB', (224, 224), (255, 255, 255))
        elif os.path.exists(image):
            image = Image.open(image)
        else:
            image = decode_base64_image(image)
            if image is None:
                return Image.new('RGB', (224, 224), (255, 255, 255))

    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')

    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # 部分 processor（如 Glm4v）要求边长 >= patch 因子 28，遇超小图直接抛错（Qwen 会自动放大）
    w, h = image.size
    if min(w, h) < 28:
        scale = 28 / min(w, h)
        image = image.resize(
            (max(int(w * scale + 0.5), 28), max(int(h * scale + 0.5), 28)), Image.LANCZOS
        )
    return image


def count_thinking_tokens(text: str, tokenizer=None) -> int:
    """Count actual tokens in thinking block using tokenizer."""
    think_end = text.find("</think>")
    if think_end != -1:
        thinking_text = text[:think_end]
    else:
        thinking_text = text

    if tokenizer is not None:
        return len(tokenizer.encode(thinking_text))
    # Fallback: rough estimate (1 word ≈ 1.3 tokens)
    return int(len(thinking_text.split()) * 1.3)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VIG evaluation (modular, per-benchmark answer checking)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["wemath", "mmmu", "mathvision",
                                "dynamath", "papo_mmk12", "geo3k"])
    parser.add_argument("--output_dir", type=str, default="output/eval_v3_base")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--cache_dir", type=str,
                       default="./hf_cache/datasets")
    parser.add_argument("--judge_mode", type=str, choices=["rule", "api", "both"],
                       default="rule",
                       help="Scoring mode: 'rule' (default) for rule-based matching, "
                            "'api' for LLM-as-judge (doubao-seed-2.0-pro), "
                            "'both' for running both and reporting both results")
    parser.add_argument("--judge_workers", type=int, default=64,
                       help="Number of parallel workers for API judge (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0.0 = greedy default; e.g. 0.6 with --seed for multi-seed sampling eval)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Sampling seed (one seed per run; repeat with different seeds for mean±std)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="top-p (GLM-4.1V official decoding uses 0.6)")
    parser.add_argument("--top_k", type=int, default=-1,
                       help="top-k (GLM-4.1V official decoding uses 2; -1 = disabled)")
    parser.add_argument("--system_prompt", type=str, choices=list(SYSTEM_PROMPT_MAP.keys()),
                       default="default",
                       help="System prompt variant ('concise_visual' = A5 training-free prompt baseline)")
    parser.add_argument("--no_image", action="store_true",
                       help="Evaluate WITHOUT images (text-only): for question answerability stratification")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Print config
    print(f"\n{'='*60}")
    print(f"VIG Evaluation")
    print(f"{'='*60}")
    print(f"Model:    {args.model_path}")
    print(f"Datasets: {args.datasets}")
    print(f"Judge:    {args.judge_mode}")
    print(f"TP:       {args.tensor_parallel_size}")
    print(f"Output:   {args.output_dir}")
    print(f"{'='*60}\n")

    # Initialize API judge if needed
    api_judge = None
    if args.judge_mode in ("api", "both"):
        print("[Judge] Initializing API judge (doubao-seed-2.0-pro)...")
        api_judge = ApiJudge(workers=args.judge_workers)

    # Load vLLM model
    print("[Model] Loading vLLM engine...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 7},
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(n=1, temperature=args.temperature, max_tokens=args.max_tokens,
                                     seed=args.seed, top_p=args.top_p, top_k=args.top_k)

    all_metrics = {}

    for ds_name in args.datasets:
        if ds_name not in DATASET_REGISTRY:
            print(f"  WARNING: Unknown dataset '{ds_name}', skipping.")
            continue

        ds_module = DATASET_REGISTRY[ds_name]

        print(f"\n{'='*60}")
        print(f"[Eval] {ds_name}")
        print(f"{'='*60}")

        # Step 1: Load dataset using module's loader
        items = ds_module.load_items(args.cache_dir)
        if not items:
            print("  No valid samples, skipping.")
            continue

        # Step 2: Build vLLM prompts (shared logic)
        vllm_inputs = []
        for item in items:
            img = item["image"]
            img = resize_image(img)

            if args.no_image:
                user_content = [{"type": "text", "text": item["question"]}]
            else:
                # 用标准 {"type": "image"}：Qwen3-VL 模板对 image/image_url 输出完全一致，
                # 而 InternVL3-hf / gemma3 模板只识别 "image"（"image_url" 会被静默丢弃，
                # 导致 prompt 无图像占位符、vLLM 多模态输入报错）。跨架构安全。
                user_content = [
                    {"type": "image"},
                    {"type": "text", "text": item["question"]},
                ]
            conv = [
                {"role": "system", "content": SYSTEM_PROMPT_MAP[args.system_prompt]},
                {"role": "user", "content": user_content},
            ]
            prompt_text = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            entry = {"prompt": prompt_text}
            if not args.no_image:
                entry["multi_modal_data"] = {"image": img}
            vllm_inputs.append(entry)

        # Step 3: Generate responses (vLLM continuous batching)
        print(f"  Generating {len(vllm_inputs)} responses...")
        outputs = llm.generate(vllm_inputs, sampling_params)
        all_responses = [o.outputs[0].text for o in outputs]

        # Step 4: Evaluate using module's extract + check functions
        correct = 0
        correct_api = 0
        total_tlen = 0
        details = []

        for item, response in zip(items, all_responses):
            tlen = count_thinking_tokens(response, tokenizer)
            total_tlen += tlen

            is_mc = item.get("is_mc", True)

            # Use dataset-specific answer extraction
            predicted = ds_module.extract_answer(response, is_mc=is_mc)

            # Rule-based correctness check
            is_correct_rule = False
            if args.judge_mode in ("rule", "both"):
                check_kwargs = {}
                if ds_name == "mathvision":
                    check_kwargs["is_mc"] = is_mc
                    check_kwargs["options"] = item.get("options", [])
                elif ds_name in ("r1_onevision", "mmmu", "realworldqa"):
                    check_kwargs["is_mc"] = is_mc
                elif ds_name == "ocrbench":
                    check_kwargs["dataset_name"] = item.get("dataset_name", "")
                    check_kwargs["question_type"] = item.get("question_type", "")
                elif ds_name == "dynamath":
                    check_kwargs["is_mc"] = is_mc
                    check_kwargs["answer_type"] = item.get("answer_type", "float")

                is_correct_rule = ds_module.check_correct(predicted, item["answer"], **check_kwargs)

            # API-based correctness check (deferred to batch)
            is_correct_api = False  # placeholder, filled in batch below

            # For rule or both mode, count rule-based correct
            if args.judge_mode == "rule":
                is_correct = is_correct_rule
            else:
                is_correct = is_correct_rule  # temporary, API fills in later

            if is_correct_rule:
                correct += 1

            details.append({
                "question": item["question"],
                "gold": item["answer"],
                "pred": predicted,
                "response": response,
                "correct": is_correct_rule,
                "tlen": tlen,
                "is_mc": is_mc,
            })

        # API judge batch (if needed)
        if args.judge_mode in ("api", "both") and api_judge is not None:
            print(f"  Running API judge on {len(details)} samples (workers={args.judge_workers})...")
            judge_items = [
                {
                    "question": d["question"],
                    "predicted": d["pred"],
                    "ground_truth": d["gold"],
                    "is_mc": d["is_mc"],
                }
                for d in details
            ]
            api_results = api_judge.check_batch(judge_items)

            # Update details with API results
            for i, api_correct in enumerate(api_results):
                details[i]["correct_api"] = api_correct
                if api_correct:
                    correct_api += 1

            # If judge_mode == "api", override the primary correct field
            if args.judge_mode == "api":
                correct = correct_api
                for d in details:
                    d["correct"] = d.get("correct_api", False)

        # Step 5: Compute metrics
        total = len(items)
        acc = correct / total * 100 if total > 0 else 0
        avg_tlen = total_tlen / total if total > 0 else 0
        eff = acc / avg_tlen * 1000 if avg_tlen > 0 else 0

        all_metrics[ds_name] = {
            "ACC": round(acc, 2),
            "TLen": round(avg_tlen, 1),
            "Eff": round(eff, 2),
            "num_samples": total,
            "num_correct": correct,
        }

        # Add API metrics if applicable
        if args.judge_mode in ("api", "both"):
            acc_api = correct_api / total * 100 if total > 0 else 0
            eff_api = acc_api / avg_tlen * 1000 if avg_tlen > 0 else 0
            all_metrics[ds_name]["ACC_api"] = round(acc_api, 2)
            all_metrics[ds_name]["Eff_api"] = round(eff_api, 2)
            all_metrics[ds_name]["num_correct_api"] = correct_api

        if args.judge_mode == "both":
            print(f"\n  >>> {ds_name}: ACC_rule={acc:.1f}% ACC_api={acc_api:.1f}% "
                  f"({correct}/{correct_api}/{total}), TLen={avg_tlen:.0f}")
        else:
            print(f"\n  >>> {ds_name}: ACC={acc:.1f}% ({correct}/{total}), "
                  f"TLen={avg_tlen:.0f}, Eff={eff:.2f}")

        # Save details as jsonl
        with open(output_path / f"{ds_name}_details.jsonl", 'w') as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {args.model_path}")
    print(f"  Judge: {args.judge_mode}")
    print(f"{'='*60}")

    if args.judge_mode == "both":
        print(f"  {'Dataset':<18} {'ACC_rule':>9} {'ACC_api':>9} {'TLen':>8} {'N':>6}")
        print(f"  {'-'*56}")
        for name, m in all_metrics.items():
            print(f"  {name:<18} {m['ACC']:>8.1f}% {m.get('ACC_api', 0):>8.1f}% "
                  f"{m['TLen']:>7.0f} {m['num_samples']:>6}")
        print(f"  {'-'*56}")
        if all_metrics:
            avg_acc = sum(m['ACC'] for m in all_metrics.values()) / len(all_metrics)
            avg_acc_api = sum(m.get('ACC_api', 0) for m in all_metrics.values()) / len(all_metrics)
            avg_tlen = sum(m['TLen'] for m in all_metrics.values()) / len(all_metrics)
            print(f"  {'AVERAGE':<18} {avg_acc:>8.1f}% {avg_acc_api:>8.1f}% {avg_tlen:>7.0f}")
    else:
        print(f"  {'Dataset':<18} {'ACC':>8} {'TLen':>8} {'Eff':>8} {'N':>6}")
        print(f"  {'-'*52}")
        for name, m in all_metrics.items():
            acc_key = 'ACC_api' if args.judge_mode == "api" and 'ACC_api' in m else 'ACC'
            eff_key = 'Eff_api' if args.judge_mode == "api" and 'Eff_api' in m else 'Eff'
            print(f"  {name:<18} {m[acc_key]:>7.1f}% {m['TLen']:>7.0f} "
                  f"{m[eff_key]:>7.2f} {m['num_samples']:>6}")
        print(f"  {'-'*52}")
        if all_metrics:
            acc_key = 'ACC_api' if args.judge_mode == "api" and 'ACC_api' in list(all_metrics.values())[0] else 'ACC'
            eff_key = 'Eff_api' if args.judge_mode == "api" and 'Eff_api' in list(all_metrics.values())[0] else 'Eff'
            avg_acc = sum(m.get(acc_key, m['ACC']) for m in all_metrics.values()) / len(all_metrics)
            avg_tlen = sum(m['TLen'] for m in all_metrics.values()) / len(all_metrics)
            avg_eff = sum(m.get(eff_key, m['Eff']) for m in all_metrics.values()) / len(all_metrics)
            print(f"  {'AVERAGE':<18} {avg_acc:>7.1f}% {avg_tlen:>7.0f} {avg_eff:>7.2f}")
    print(f"{'='*60}\n")

    with open(output_path / "summary.json", 'w') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}/")


if __name__ == "__main__":
    main()
