"""Score the sentences of one reasoning chain by VIG.

This is the smallest end-to-end use of the reward outside RL training: load a
policy, generate a chain for one image-question pair, then report the
sentence-level visual information gain that VIG would use as its reward signal.

Usage:
    python examples/score_chain.py --model Qwen/Qwen3-VL-8B-Thinking \
        --image path/to/figure.png --question "What is the length of AB in cm?"
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vig import VIGCompressor  # noqa: E402


SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. "
    "Think step by step, focusing on information directly relevant to solving "
    "the problem. Format your response as:\n"
    "<think>\nYour concise step-by-step reasoning\n</think>\n"
    "<answer>\nYour final answer\n</answer>"
)


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if len(p.strip()) > 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--vision_span_mode", default="qwen3vl",
                    help="qwen3vl | internvl | gemma3 | glm4v | auto")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": args.question}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False)
    completion = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
    think = completion.split("</think>")[0].replace("<think>", "").strip()
    sentences = split_sentences(think)
    if not sentences:
        print("No <think> block was produced; nothing to score.")
        return

    compressor = VIGCompressor(model, processor.tokenizer, processor=processor,
                               vision_span_mode=args.vision_span_mode)
    scores = compressor.compute_vig(sentences, args.question, image)

    print(f"\n{'VIG':>8}  {'H_vis':>7}  {'H_txt':>7}   Sentence")
    print("-" * 78)
    for s in scores:
        print(f"{s.VIG:+8.3f}  {s.H_vis:7.3f}  {s.H_txt:7.3f}   {s.step[:52]}")
    mean_vig = sum(s.VIG for s in scores) / len(scores)
    print("-" * 78)
    print(f"chain-level R_VIG = clip(mean, -1, 1) = "
          f"{max(-1.0, min(1.0, mean_vig)):+.4f}   ({len(scores)} sentences)")


if __name__ == "__main__":
    main()
