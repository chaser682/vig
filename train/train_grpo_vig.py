"""
GRPO 训练主脚本（直接 RL 阶段，不含 SFT 冷启动）。

模型：Qwen3-VL-8B-Thinking（全参数训练 + DeepSpeed Zero2）
奖励：R_format + R_acc + R_VIG（不含 R_len）
框架：trl GRPOTrainer

设计原则
--------
• 奖励逻辑全部由 train/rewards.py 提供，本文件只负责组装 Trainer。
• VIG 奖励使用独立的 VIGCompressor 实例（与 GRPOTrainer 管理的训练模型
  完全隔离），彻底规避 vLLM colocate 模式下权重同步导致的前向传播崩溃。
• VIGCompressor 内部模型使用 attn_implementation="flash_attention_2" 绕开 SDPA GQA
  在某些 PyTorch 版本下的输出形状 bug（(N,0) attention），同时获得 FA2 加速。

用法：
    accelerate launch --config_file configs/accelerate_zero2.yaml \\
        train/train_grpo_vig.py --config configs/grpo_vig_token.yaml
"""

import os
import sys
import argparse
from typing import List, Optional

import torch
import yaml
from datasets import Dataset as HFDataset

# 项目根路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# 奖励函数（全部从 rewards.py 导入，与训练模型无关）
# ---------------------------------------------------------------------------
from train.rewards import (
    grpo_format_reward,
    grpo_accuracy_reward,
    make_vig_reward_func,
    make_soft_overlong_reward_func,
)


# ---------------------------------------------------------------------------
# 数据集构建
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. "
    "Think step by step, focusing on information directly relevant to solving the problem. "
    "Avoid redundant or generic statements. "
    "Format your response as:\n"
    "<think>\nYour concise step-by-step reasoning\n</think>\n"
    "<answer>\nYour final answer\n</answer>"
)

# VIG-aware System Prompt (Layer 3)：
# 明确引导模型将视觉信息与文本推理剥离，仅在图像提供关键信息时深入分析，
# 文本逻辑步骤尽量精简，避免重复问题表述。
SYSTEM_PROMPT_VIG_AWARE = (
    "You are a helpful visual reasoning assistant. "
    "Focus your reasoning on steps where the image provides CRITICAL information "
    "that you could NOT infer from the question text alone. "
    "For steps that only involve text-based logic or arithmetic, be maximally brief. "
    "Do NOT repeat or paraphrase the problem statement in your thinking. "
    "Format your response as:\n"
    "<think>\nYour concise step-by-step reasoning\n</think>\n"
    "<answer>\nYour final answer\n</answer>"
)

_SYSTEM_PROMPT_MAP = {
    "default": SYSTEM_PROMPT,
    "vig_aware": SYSTEM_PROMPT_VIG_AWARE,
}


def build_dataset(
    cache_dir: str,
    sources: List[str],
    max_per_source: Optional[int] = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> HFDataset:
    """
    构建 trl GRPOTrainer 所需的 HuggingFace Dataset。

    trl 要求 dataset 包含 "prompt" 字段（conversation 格式）。
    其余字段（如 "answer"）会作为 reward_kwargs 传给 reward_funcs。
    """
    from datasets import load_dataset

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
                "image": item.get("image", None),
                "source": "mmstar",
                "choices": [],  # MMStar options already embedded in question
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
                "image": item.get("decoded_image", item.get("image", None)),
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
                "image": item.get("image", None),
                "source": "logicvista",
                "choices": choices_list,
            })
        print(f"  LogicVista: {len(ds)} samples")

    print(f"[Data] Total: {len(all_samples)} samples")

    # --- 构建 prompt（conversation 格式）---
    samples_for_hf = []
    for sample in all_samples:
        prompt = [
            {"role": "system", "content": system_prompt},
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


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VIG GRPO Training")
    parser.add_argument("--config", type=str, default="configs/grpo_vig_token.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from (e.g. output/grpo_vig_token/checkpoint-500)")
    args = parser.parse_args()

    # --- 加载配置 ---
    with open(os.path.join(ROOT, args.config)) as f:
        cfg = yaml.safe_load(f)

    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.max_per_source:
        cfg["max_per_source"] = args.max_per_source
    if args.resume_from_checkpoint:
        cfg["resume_from_checkpoint"] = args.resume_from_checkpoint

    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # --- GRPOConfig ---
    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
        learning_rate=cfg.get("learning_rate", 1e-6),
        warmup_steps=cfg.get("warmup_steps", 10),
        max_steps=cfg.get("max_steps", 500),
        num_generations=cfg.get("group_size", 8),
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 0.95),
        max_completion_length=cfg.get("max_new_tokens", 4096),
        beta=cfg.get("kl_coef", 0.02),
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 100),
        bf16=True,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        seed=cfg.get("seed", 42),
        report_to=cfg.get("report_to", "tensorboard"),
        remove_unused_columns=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        log_completions=True,
        # GRPO-specific — 可通过 yaml 覆盖，默认保持原来行为
        num_iterations=1,
        loss_type=cfg.get("loss_type", "grpo"),
        scale_rewards=cfg.get("scale_rewards", "group"),
        epsilon=cfg.get("epsilon", 0.2),
        epsilon_high=cfg.get("epsilon_high", None),
        mask_truncated_completions=cfg.get("mask_truncated_completions", False),
        # vLLM colocate（与 baselines 保持一致）
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=False,
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.3),
        vllm_max_model_length=32768,
        vllm_tensor_parallel_size=1,
        vllm_importance_sampling_correction=False,
    )

    # --- 加载训练模型和 processor ---
    model_path = cfg["model_name_or_path"]
    print(f"[Model] Loading training model: {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # InternVL 等动态分块模型：强制单块，保证 trl 训练前向的 token/特征一致（cfg: force_single_tile）
    if cfg.get("force_single_tile", False):
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            if hasattr(ip, "crop_to_patches"):
                ip.crop_to_patches = False
            if hasattr(ip, "max_patches"):
                ip.max_patches = 1
            print("  [Processor] force_single_tile: crop_to_patches=False, max_patches=1")
    # ZeRO-3 下必须先让 accelerate 的 init hook 接管参数分片，
    # 用 device_map=None + low_cpu_mem_usage=True 避免每 rank 先完整加载再 scatter
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # --- VIG 奖励函数 ---
    # R_VIG = avg_KeepScore = avg(VIG + μ*VIG)，clip 到 (-1, 1)
    # 直接用训练模型做 forward（eval + no_grad），不需要独立模型。
    # 有图 forward：processor 编码 image+question+生成文本 → logits_vis
    # 无图 forward：去掉 vision token → logits_txt
    # VIG = H_txt - H_vis，按步骤切分取平均
    from vig.compute_vig import VIGCompressor
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    vig_device = f"cuda:{local_rank}"

    w_format = cfg.get("w_format", 0.1)
    w_acc    = cfg.get("w_acc",    1.0)
    w_vig   = cfg.get("w_vig",   1.0)   # paper setting

    if w_vig > 0:
        vig_compressor = VIGCompressor(
            model=model,
            tokenizer=processor.tokenizer,
            processor=processor,
            device=vig_device,
            comparison_mode=cfg.get("vig_comparison_mode", "no_image"),
            mask_ratio=cfg.get("vig_mask_ratio", 0.5),
            aggregation_mode=cfg.get("vig_aggregation_mode", "mean"),
            split_high_variance=cfg.get("vig_split_high_variance", False),
            variance_threshold=cfg.get("vig_variance_threshold", 2.0),
            vision_span_mode=cfg.get("vision_span_mode", "qwen3vl"),
        )
        vig_fn = make_vig_reward_func(
            vig_compressor=vig_compressor,
            lambda_=cfg.get("vig_lambda", 1.0),
            mu=cfg.get("vig_mu", 0.5),
            baseline=cfg.get("vig_baseline", 0.5),
            max_steps_per_sample=cfg.get("vig_max_steps", None),
            score_mode=cfg.get("vig_score_mode", "vig_only"),   # paper setting
            gamma=cfg.get("vig_gamma", 0.0),
            use_position_weight=cfg.get("vig_position_weight", False),
            use_zscore_norm=cfg.get("vig_zscore_norm", False),
            step_split_mode=cfg.get("vig_step_split_mode", "token"),   # paper setting
        )
        print(f"  [Reward] VIG: score_mode={cfg.get('vig_score_mode', 'vig_only')}, "
              f"comparison_mode={cfg.get('vig_comparison_mode', 'no_image')}, "
              f"vision_span_mode={cfg.get('vision_span_mode', 'qwen3vl')}, "
              f"aggregation_mode={cfg.get('vig_aggregation_mode', 'mean')}, "
              f"step_split_mode={cfg.get('vig_step_split_mode', 'token')}, "
              f"gamma={cfg.get('vig_gamma', 0.0)}, "
              f"position_weight={cfg.get('vig_position_weight', False)}, "
              f"zscore_norm={cfg.get('vig_zscore_norm', False)}, "
              f"split_high_variance={cfg.get('vig_split_high_variance', False)}, "
              f"R = clip(avg_KeepScore - {cfg.get('vig_baseline', 0.5)}, -1, 1)")
    else:
        # w_vig=0：不使用 VIG 奖励，用零奖励占位（GRPOTrainer 需要 3 个 reward_funcs 对齐权重）
        def vig_fn(completions, **kwargs):
            return [0.0] * len(completions)
        print("  [Reward] VIG: disabled (w_vig=0.0), using zero placeholder")

    # --- 奖励函数列表 ---
    reward_funcs = [grpo_format_reward, grpo_accuracy_reward, vig_fn]
    reward_weights = [w_format, w_acc, w_vig]

    # --- Soft Overlong 惩罚（可选，DAPO 配套）---
    w_soft_overlong = cfg.get("w_soft_overlong", 0.0)
    if w_soft_overlong > 0:
        soft_overlong_fn = make_soft_overlong_reward_func(
            min_len=cfg.get("soft_overlong_min_len", 1024),
            max_len=cfg.get("soft_overlong_max_len", 2048),
        )
        reward_funcs.append(soft_overlong_fn)
        reward_weights.append(w_soft_overlong)
        print(f"  [Reward] SoftOverlong: min_len={cfg.get('soft_overlong_min_len', 1024)}, "
              f"max_len={cfg.get('soft_overlong_max_len', 2048)}, w={w_soft_overlong}")
    else:
        print("  [Reward] SoftOverlong: disabled")

    grpo_config.reward_weights = reward_weights

    # --- 系统提示词 ---
    sp_type = cfg.get("system_prompt_type", "default")
    active_system_prompt = _SYSTEM_PROMPT_MAP.get(sp_type, SYSTEM_PROMPT)
    print(f"  [SP] system_prompt_type={sp_type!r}")

    # --- 构建数据集 ---
    dataset = build_dataset(
        cache_dir=cfg.get("data_cache_dir"),
        sources=cfg.get("sources", ["mmstar", "mathvista", "logicvista"]),
        max_per_source=cfg.get("max_per_source"),
        system_prompt=active_system_prompt,
    )

    # --- 创建 Trainer ---
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=processor,
    )

    # --- 打印配置摘要 ---
    print("=" * 60)
    print("[GRPO] Training Configuration:")
    print(f"  Model:            {model_path}")
    print(f"  Mode:             Full-parameter (no LoRA)")
    print(f"  System Prompt:    {sp_type}")
    print(f"  Max steps:        {cfg.get('max_steps', 500)}")
    print(f"  Group size G:     {cfg.get('group_size', 8)}")
    print(f"  Batch/GPU:        {cfg.get('per_device_train_batch_size', 2)}")
    print(f"  Grad accum:       {cfg.get('gradient_accumulation_steps', 2)}")
    print(f"  Rewards:          format({w_format}) + acc({w_acc}) + vig({w_vig})"
          + (f" + soft_overlong({w_soft_overlong})" if w_soft_overlong > 0 else ""))
    print(f"  Loss type:        {cfg.get('loss_type', 'grpo')}, "
          f"epsilon={cfg.get('epsilon', 0.2)}, epsilon_high={cfg.get('epsilon_high', None)}, "
          f"mask_truncated={cfg.get('mask_truncated_completions', False)}")
    print(f"  VIG mode:        score_mode={cfg.get('vig_score_mode', 'full')}, "
          f"comparison_mode={cfg.get('vig_comparison_mode', 'no_image')}, "
          f"aggregation_mode={cfg.get('vig_aggregation_mode', 'mean')}, "
          f"step_split_mode={cfg.get('vig_step_split_mode', 'newline')}, "
          f"gamma={cfg.get('vig_gamma', 0.0)}, "
          f"position_weight={cfg.get('vig_position_weight', False)}, "
          f"zscore_norm={cfg.get('vig_zscore_norm', False)}, "
          f"split_high_variance={cfg.get('vig_split_high_variance', False)}, "
          f"baseline={cfg.get('vig_baseline', 0.5)}")
    print(f"  Dataset size:     {len(dataset)}")
    if cfg.get("resume_from_checkpoint"):
        print(f"  Resume from:      {cfg['resume_from_checkpoint']}")
    print("=" * 60)

    resume_ckpt = cfg.get("resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # --- 保存 ---
    final_path = os.path.join(cfg["output_dir"], "final")
    trainer.save_model(final_path)
    if processor is not None:
        processor.save_pretrained(final_path)
    print(f"\n[GRPO] Training done. Model saved to {final_path}")


if __name__ == "__main__":
    main()
