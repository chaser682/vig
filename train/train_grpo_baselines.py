"""
GRPO 对比方法训练脚本（三种方法共用）

严格参照官方 trl grpo_vlm.py 写法：
  - accelerate launch --config_file configs/accelerate_zero3.yaml
  - 模型传路径字符串，GRPOTrainer 内部加载（ZeRO-3 init_flag 生效）
  - use_vllm + vllm_mode colocate + vllm_enable_sleep_mode（sleep模式：生成时申请KV，训练时释放）

方法 1 - Pure GRPO (grpo_pure.yaml):   R = R_format + R_acc
方法 2 - L1-LCPO (grpo_l1.yaml):       R = R_format + R_acc + R_length
方法 3 - ThinkPrune (grpo_thinkprune.yaml): 截断到budget后判断正确性

数据加载与 acc 计算与 train_grpo_vig.py 完全对齐：
  - build_dataset 保存 choices 字段，供 MC 字母↔选项文本双向匹配
  - format / accuracy reward 统一使用 train.rewards 中的 grpo_format_reward /
    grpo_accuracy_reward / compute_accuracy_reward，与 VIG 训练脚本行为一致

用法：
    accelerate launch --config_file configs/accelerate_zero3.yaml \\
        train/train_grpo_baselines.py --config configs/grpo_pure.yaml
"""

import os
import sys
import re
import random
import argparse
from typing import Optional, List

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# 统一使用 rewards.py 中的 format / accuracy 奖励（与 train_grpo_vig.py 对齐）
# ---------------------------------------------------------------------------
from train.rewards import grpo_format_reward, grpo_accuracy_reward, compute_accuracy_reward


# ============================================================
# 通用工具
# ============================================================

def _get_text(completion) -> str:
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion or ""


def _ensure_rgb(image):
    if image is None:
        return None
    try:
        if hasattr(image, "mode") and image.mode != "RGB":
            return image.convert("RGB")
        return image
    except Exception:
        return image


# ============================================================
# 奖励函数（仅保留 L1 长度惩罚 和 ThinkPrune 截断逻辑；
# format / accuracy 统一从 rewards.py 导入）
# ============================================================

def make_l1_reward_funcs(alpha: float, budget_min: int, budget_max: int,
                          mode: str = "exact", tokenizer=None):
    """
    L1-LCPO 奖励：
      - acc_fn   使用 compute_accuracy_reward（与 VIG 对齐，含 MC 双向匹配）
      - len_fn   线性长度惩罚，budget 从 prompt 的 system 消息中解析
    """
    def l1_accuracy_reward_func(prompts, completions, **kwargs) -> list:
        ground_truths = kwargs.get("answer", [])
        choices_list = kwargs.get("choices", [])
        rewards = []
        for i, completion in enumerate(completions):
            text = _get_text(completion)
            if not text.strip():
                rewards.append(0.0)
                continue
            gold = ground_truths[i] if i < len(ground_truths) else ""
            choices = choices_list[i] if i < len(choices_list) else []
            if not choices:
                choices = None
            rewards.append(compute_accuracy_reward(text, gold, choices=choices))
        return rewards

    def l1_length_reward_func(prompts, completions, **kwargs) -> list:
        import re as _re
        rewards = []
        for i, completion in enumerate(completions):
            text = _get_text(completion)
            if not text.strip():
                rewards.append(0.0)
                continue
            # 从 prompt 的 system 消息里解析 budget（与模型实际看到的保持一致）
            budget = budget_min  # fallback
            prompt = prompts[i] if i < len(prompts) else []
            for msg in (prompt if isinstance(prompt, list) else []):
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") for c in content if isinstance(c, dict)
                        )
                    m = _re.search(r"Think for exactly (\d+) tokens", content)
                    if m:
                        budget = int(m.group(1))
                    break
            approx_tokens = (
                len(tokenizer.encode(text, add_special_tokens=False))
                if tokenizer is not None
                else len(text.split())
            )
            if mode == "exact":
                penalty = -abs(approx_tokens - budget) * alpha
            else:
                penalty = -max(approx_tokens - budget, 0) * alpha
            rewards.append(max(-3.0, min(0.0, penalty)))
        return rewards

    return l1_accuracy_reward_func, l1_length_reward_func


def make_thinkprune_reward_funcs(budget: int, tokenizer=None):
    """
    ThinkPrune 奖励：截断到 budget 后判断正确性。
    acc 使用 compute_accuracy_reward（与 VIG 对齐，含 MC 双向匹配）。
    """
    def truncate_to_budget(text: str) -> str:
        if tokenizer is not None:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= budget:
                return text
            return tokenizer.decode(ids[:budget], skip_special_tokens=True)
        words = text.split()
        return " ".join(words[:budget]) if len(words) > budget else text

    def thinkprune_accuracy_reward_func(prompts, completions, **kwargs) -> list:
        ground_truths = kwargs.get("answer", [])
        choices_list = kwargs.get("choices", [])
        rewards = []
        for i, completion in enumerate(completions):
            text = _get_text(completion)
            truncated = truncate_to_budget(text)
            if not truncated.strip():
                rewards.append(0.0)
                continue
            if not re.search(r"<answer>.*?</answer>", truncated, re.DOTALL):
                rewards.append(0.0)
                continue
            gold = ground_truths[i] if i < len(ground_truths) else ""
            choices = choices_list[i] if i < len(choices_list) else []
            if not choices:
                choices = None
            rewards.append(compute_accuracy_reward(truncated, gold, choices=choices))
        return rewards

    return thinkprune_accuracy_reward_func


# ============================================================
# System prompts
# ============================================================

SYSTEM_PROMPT_BASE = (
    "You are a helpful visual reasoning assistant. "
    "Think step by step, focusing on information directly relevant to solving the problem. "
    "Avoid redundant or generic statements. "
    "Format your response as:\n"
    "<think>\nYour concise step-by-step reasoning\n</think>\n"
    "<answer>\nYour final answer\n</answer>"
)

SYSTEM_PROMPT_L1 = SYSTEM_PROMPT_BASE + "\nThink for exactly {budget} tokens."

SYSTEM_PROMPT_THINKPRUNE = SYSTEM_PROMPT_BASE + "\nOutput must be within {budget} tokens."


# ============================================================
# 数据集（与 train_grpo_vig.py 完全对齐：保留 choices 字段）
# ============================================================

def build_dataset(
    cache_dir: str,
    sources: List[str],
    max_per_source: Optional[int],
    reward_type: str,
    l1_budget_min: int = 200,
    l1_budget_max: int = 2000,
    thinkprune_budget: int = 2048,
):
    from datasets import load_dataset, Dataset as HFDataset

    all_samples = []

    # --- MMStar ---
    if "mmstar" in sources:
        print("[Data] Loading MMStar ...")
        ds = load_dataset("Lin-Chen/MMStar", split="val", cache_dir=cache_dir)
        if max_per_source:
            ds = ds.select(range(min(max_per_source, len(ds))))
        for item in ds:
            all_samples.append({
                "question": item.get("question", ""),
                "answer": str(item.get("answer", "")),
                "image": _ensure_rgb(item.get("image")),
                "source": "mmstar",
                "choices": [],   # MMStar 选项已内嵌在 question 中
            })
        print(f"  MMStar: {len(ds)} samples")

    # --- MathVista ---
    if "mathvista" in sources:
        print("[Data] Loading MathVista ...")
        ds = load_dataset("AI4Math/MathVista", split="testmini", cache_dir=cache_dir)
        if max_per_source:
            ds = ds.select(range(min(max_per_source, len(ds))))
        for item in ds:
            question = item.get("question", "")
            choices = item.get("choices", None)
            choices_list = []
            if choices and isinstance(choices, (list, tuple)):
                choices_list = [str(c) for c in choices]
                labels = "ABCDEFGH"
                opts_str = " ".join(f"({labels[i]}) {o}" for i, o in enumerate(choices))
                question = f"{question}\nOptions: {opts_str}"
            all_samples.append({
                "question": question,
                "answer": str(item.get("answer", "")),
                "image": _ensure_rgb(item.get("decoded_image", item.get("image"))),
                "source": "mathvista",
                "choices": choices_list,
            })
        print(f"  MathVista: {len(ds)} samples")

    # --- LogicVista ---
    if "logicvista" in sources:
        print("[Data] Loading LogicVista ...")
        ds = load_dataset("lscpku/LogicVista", split="test", cache_dir=cache_dir)
        if max_per_source:
            ds = ds.select(range(min(max_per_source, len(ds))))
        for item in ds:
            question = item.get("question", item.get("problem", ""))
            choices = item.get("choices", item.get("options", None))
            choices_list = []
            if choices and isinstance(choices, (list, tuple)):
                choices_list = [str(c) for c in choices]
                labels = "ABCDEFGH"
                opts_str = " ".join(f"({labels[i]}) {o}" for i, o in enumerate(choices))
                question = f"{question}\nOptions: {opts_str}"
            all_samples.append({
                "question": question,
                "answer": str(item.get("answer", item.get("label", ""))),
                "image": _ensure_rgb(item.get("image")),
                "source": "logicvista",
                "choices": choices_list,
            })
        print(f"  LogicVista: {len(ds)} samples")

    print(f"[Data] Total: {len(all_samples)} samples")

    # --- 构建 HF Dataset（prompt 用 conversation 格式，保留 choices 字段）---
    samples_for_hf = []
    for sample in all_samples:
        if reward_type == "l1":
            # 每条样本随机采样 budget 并写入 system prompt；reward 函数从 prompt 解析该值
            budget = random.randint(l1_budget_min, l1_budget_max)
            sys_prompt = SYSTEM_PROMPT_L1.format(budget=budget)
        elif reward_type == "thinkprune":
            sys_prompt = SYSTEM_PROMPT_THINKPRUNE.format(budget=thinkprune_budget)
        else:
            sys_prompt = SYSTEM_PROMPT_BASE

        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": sample["question"]},
        ]
        item = {
            "prompt": prompt,
            "answer": sample["answer"],
            "choices": sample.get("choices", []),
        }
        if sample["image"] is not None:
            item["image"] = sample["image"]
        samples_for_hf.append(item)

    return HFDataset.from_list(samples_for_hf)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from (e.g. output/grpo_pure_4b/checkpoint-400)")
    args = parser.parse_args()

    with open(os.path.join(ROOT, args.config)) as f:
        cfg = yaml.safe_load(f)

    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.max_per_source:
        cfg["max_per_source"] = args.max_per_source
    if args.resume_from_checkpoint:
        cfg["resume_from_checkpoint"] = args.resume_from_checkpoint

    reward_type = cfg.get("reward_type", "pure")
    model_path = cfg["model_name_or_path"]
    max_new_tokens = cfg.get("max_new_tokens", 2048)

    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoProcessor

    # ---- GRPOConfig（严格对齐官方 grpo_vlm.py 参数风格）----
    training_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        # 训练超参
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.get("learning_rate", 1e-6),
        warmup_steps=cfg.get("warmup_steps", 10),
        max_steps=cfg.get("max_steps", 500),
        bf16=True,
        max_grad_norm=1.0,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        # 不要 use_reentrant=True（ZeRO-3 下会死锁）
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # GRPO 专属
        num_generations=cfg.get("group_size", 8),
        max_completion_length=max_new_tokens,
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 0.95),
        beta=cfg.get("kl_coef", 0.04),
        epsilon=cfg.get("epsilon", 0.2),
        epsilon_high=cfg.get("epsilon_high", 0.2),
        loss_type="grpo",
        scale_rewards="group",
        num_iterations=1,
        # 日志/保存
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=cfg.get("save_steps", 100),
        report_to=cfg.get("report_to", "tensorboard"),
        log_completions=True,
        # 杂项
        seed=cfg.get("seed", 42),
        remove_unused_columns=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        # 模型加载：bf16 + flash_attention_2（官方写法）
        model_init_kwargs={
            "dtype": "bfloat16",
            # flash_attention_2 需要 flash-attn 包；环境缺失时可用 cfg attn_implementation 覆盖为 sdpa
            "attn_implementation": cfg.get("attn_implementation", "flash_attention_2"),
            "trust_remote_code": True,
        },
        # vLLM：colocate，不用 sleep_mode
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=False,
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.35),
        vllm_max_model_length=32768,
        vllm_tensor_parallel_size=1,
        vllm_importance_sampling_correction=False,
    )

    # processor 只在主进程加载（ZeRO-3 下 model 由 trainer 内部初始化）
    print(f"[Setup] reward_type={reward_type}, model={model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if cfg.get("force_single_tile", False):
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            if hasattr(ip, "crop_to_patches"):
                ip.crop_to_patches = False
            if hasattr(ip, "max_patches"):
                ip.max_patches = 1
            print("  [Processor] force_single_tile: crop_to_patches=False, max_patches=1")

    # ---- 奖励函数（format/acc 统一用 rewards.py，与 VIG 对齐）----
    if reward_type == "pure":
        reward_funcs = [grpo_format_reward, grpo_accuracy_reward]
        reward_weights = [cfg.get("w_format", 0.1), cfg.get("w_acc", 1.0)]

    elif reward_type == "l1":
        acc_fn, len_fn = make_l1_reward_funcs(
            alpha=cfg.get("l1_alpha", 0.0003),
            budget_min=cfg.get("l1_budget_min", 200),
            budget_max=cfg.get("l1_budget_max", 2000),
            mode=cfg.get("l1_mode", "exact"),
            tokenizer=processor.tokenizer,
        )
        reward_funcs = [grpo_format_reward, acc_fn, len_fn]
        reward_weights = [cfg.get("w_format", 0.1), cfg.get("w_acc", 1.0), cfg.get("w_length", 1.0)]

    elif reward_type == "thinkprune":
        tp_acc_fn = make_thinkprune_reward_funcs(
            budget=cfg.get("thinkprune_budget", 2048),
            tokenizer=processor.tokenizer if cfg.get("thinkprune_truncate_at_budget", True) else None,
        )
        reward_funcs = [grpo_format_reward, tp_acc_fn]
        reward_weights = [cfg.get("w_format", 0.1), cfg.get("w_acc", 1.0)]

    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    training_args.reward_weights = reward_weights

    # ---- 数据集 ----
    dataset = build_dataset(
        cache_dir=cfg.get("data_cache_dir"),
        sources=cfg.get("sources", ["mmstar", "mathvista", "logicvista"]),
        max_per_source=cfg.get("max_per_source"),
        reward_type=reward_type,
        l1_budget_min=cfg.get("l1_budget_min", 200),
        l1_budget_max=cfg.get("l1_budget_max", 2000),
        thinkprune_budget=cfg.get("thinkprune_budget", 2048),
    )

    print("=" * 60)
    print(f"  Method:      {reward_type.upper()}")
    print(f"  Model:       {model_path}")
    print(f"  Output:      {cfg['output_dir']}")
    print(f"  Steps:       {cfg.get('max_steps', 500)}")
    print(f"  Batch×Accum: {cfg.get('per_device_train_batch_size', 1)}×{cfg.get('gradient_accumulation_steps', 1)}")
    print(f"  G (group):   {cfg.get('group_size', 8)}")
    print(f"  MaxTokens:   {max_new_tokens}")
    print(f"  Weights:     {reward_weights}")
    print(f"  Dataset:     {len(dataset)}")
    print("=" * 60)

    # ---- Trainer（官方写法：model 传路径字符串）----
    trainer = GRPOTrainer(
        model=model_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None))

    final_path = os.path.join(cfg["output_dir"], "final")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    print(f"[Done] Model saved to {final_path}")


if __name__ == "__main__":
    main()
