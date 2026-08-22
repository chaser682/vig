"""
VIGCompressor: 视觉感知步骤熵（Visual-Aware Step Entropy, VIG）计算核心。

极简高效实现（v4）
------------------
核心思路：利用 processor 编码完整对话，只做 2 次 forward：
  1. processor 编码 [user: image+question, assistant: 生成文本] → forward → logits_vis
  2. 从 input_ids 中删掉 <|vision_start|>...<|vision_end|> → forward → logits_txt
     （或消融：传入随机 mask 50% 像素的图像 → forward → logits_masked）
  3. 定位生成文本 token，按步骤边界切分 → 计算 VIG

无需额外 model，直接用训练模型（eval 模式 + no_grad）。
与 ZeRO-3 兼容（在 reward 函数中调用，不需要独立模型实例）。

comparison_mode 参数（用于消融实验）：
  "no_image"     ：原始行为，第二次 forward 去掉 vision token（纯文本）。
  "masked_image" ：第二次 forward 传入随机 mask 50% patch 的图像（H_masked 替代 H_txt）。
                   此时 VIG = H_masked - H_vis，衡量"原图相比遮挡图"的额外信息增益。
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .step_parser import StepParser


# ---------------------------------------------------------------------------
# 跨架构 vision-span 识别配置
# ---------------------------------------------------------------------------
# 每种架构在 input_ids 中的图像 token 区段结构：
#   qwen3vl : <|vision_start|> <|image_pad|>*N <|vision_end|>
#   internvl: <img> <IMG_CONTEXT>*N </img>          （InternVL3-*-hf, transformers 原生）
#   gemma3  : <start_of_image> <image_soft_token>*256 <end_of_image>
#   glm4v   : <|begin_of_image|> <|image|>*N <|end_of_image|>   （GLM-4.1V-Thinking, transformers 原生）
#   auto    : 不依赖 marker，直接按 image_token_id 连续段检测并删除
# marker 模式下删除 [start_marker, end_marker] 含两端的所有 token。
_VISION_SPAN_PRESETS = {
    # mode: (start_marker_token, end_marker_token, image_token)
    "qwen3vl":  ("<|vision_start|>", "<|vision_end|>", None),
    "internvl": ("<img>", "</img>", "<IMG_CONTEXT>"),
    "gemma3":   ("<start_of_image>", "<end_of_image>", "<image_soft_token>"),
    "glm4v":    ("<|begin_of_image|>", "<|end_of_image|>", "<|image|>"),
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class StepScore:
    step: str
    index: int
    H_vis: float         # 有图条件熵（聚合值）
    H_txt: float         # 无图条件熵（聚合值）
    VIG: float           # 视觉信息增益 = H_txt - H_vis
    H_vis_std: float = 0.0   # step 内 H_vis 的标准差（反映 step 内部不确定性分散度）
    H_vis_max: float = 0.0   # step 内 H_vis 的最大值


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class VIGCompressor:
    """
    基于 processor + model forward 计算 VIG。

    每个样本只需 2 次 forward pass：
      - 有图 forward：完整 image+question+生成文本
      - 无图 forward：去掉 vision token 后的纯文本

    无需独立模型实例，直接在 reward 函数中调用。
    """

    def __init__(
        self,
        model,
        tokenizer,
        processor=None,
        device: str = "cuda",
        min_step_tokens: int = 5,
        comparison_mode: str = "no_image",
        mask_ratio: float = 0.5,
        aggregation_mode: str = "mean",
        split_high_variance: bool = False,
        variance_threshold: float = 2.0,
        vision_span_mode: str = "qwen3vl",
        image_token_id: Optional[int] = None,
    ):
        """
        Args:
            model:               训练模型（eval + no_grad 下调用）。
            tokenizer:           对应 tokenizer。
            processor:           多模态 processor（可选）。
            device:              计算设备。
            min_step_tokens:     StepParser 最小步骤 token 数。
            comparison_mode:     "no_image"     — 标准：第二次 forward 去掉 vision token（原始行为）。
                                 "masked_image" — 消融：第二次 forward 传入随机 mask 50% patch 的图像。
            mask_ratio:          masked_image 模式下 patch 遮挡比例，默认 0.5（50%）。
            aggregation_mode:    step 内 entropy 聚合方式：
                                   "mean"      — 原始均值（默认）。
                                   "max_mean"  — 0.5*max + 0.5*mean（兼顾均值和峰值）。
                                   "max_std"   — 0.5*mean + 0.3*max + 0.2*std（加入方差项）。
            split_high_variance: 是否将高方差 step 动态切分为子段（True=精细压缩决策）。
            variance_threshold:  split_high_variance=True 时，std > threshold 才切分。
            vision_span_mode:    vision token 区段识别方式（跨架构适配）：
                                   "qwen3vl"  — <|vision_start|>...<|vision_end|>（默认，原始行为）。
                                   "internvl" — <img><IMG_CONTEXT>*N</img>（InternVL3-hf）。
                                   "gemma3"   — <start_of_image><image_soft_token>*N<end_of_image>。
                                   "glm4v"    — <|begin_of_image|><|image|>*N<|end_of_image|>（GLM-4.1V）。
                                   "auto"     — 按 image_token_id 连续段检测（无需 marker）。
            image_token_id:      "auto" 模式下的图像 token id；缺省时依次从
                                 model.config.image_token_id / image_token_index 解析。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.step_parser = StepParser(min_step_tokens=min_step_tokens)
        self.comparison_mode = comparison_mode
        self.mask_ratio = mask_ratio
        self.aggregation_mode = aggregation_mode
        self.split_high_variance = split_high_variance
        self.variance_threshold = variance_threshold

        # --- vision-span 识别（跨架构）---
        if vision_span_mode not in (*_VISION_SPAN_PRESETS, "auto"):
            raise ValueError(
                f"Unknown vision_span_mode: {vision_span_mode!r}, "
                f"expected one of {list(_VISION_SPAN_PRESETS) + ['auto']}"
            )
        self.vision_span_mode = vision_span_mode

        if vision_span_mode == "auto":
            # 不依赖 marker：直接删除所有 image_token_id（连续段）
            self.vision_start_id = None
            self.vision_end_id = None
            if image_token_id is None:
                mcfg = getattr(model, "config", None)
                image_token_id = (
                    getattr(mcfg, "image_token_id", None)
                    or getattr(mcfg, "image_token_index", None)
                )
            if image_token_id is None:
                raise ValueError(
                    "vision_span_mode='auto' 需要 image_token_id"
                    "（显式传入或 model.config.image_token_id / image_token_index）"
                )
            self.image_token_id = image_token_id
        else:
            start_tok, end_tok, img_tok = _VISION_SPAN_PRESETS[vision_span_mode]
            # 特殊 token id（qwen3vl 路径与原始行为完全一致）
            self.vision_start_id = tokenizer.convert_tokens_to_ids(start_tok)
            self.vision_end_id = tokenizer.convert_tokens_to_ids(end_tok)
            self.image_token_id = (
                tokenizer.convert_tokens_to_ids(img_tok) if img_tok else None
            )
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if (
                self.vision_start_id is None or self.vision_end_id is None
                or self.vision_start_id == unk_id or self.vision_end_id == unk_id
            ):
                raise ValueError(
                    f"vision_span_mode={vision_span_mode!r} 的 marker token "
                    f"({start_tok!r}/{end_tok!r}) 不在该 tokenizer 词表中，"
                    f"请换用正确的 mode 或 'auto'"
                )

        # 第二次 forward 之外，有图 forward 需要透传的多模态 kwargs。
        # qwen3vl 保持原 key 集合不变（回归安全）；其余架构追加 token_type_ids
        # （gemma3 processor 会返回，用于标记图像位置）。
        if vision_span_mode == "qwen3vl":
            self._mm_forward_keys = ("pixel_values", "image_grid_thw", "mm_token_type_ids")
        else:
            self._mm_forward_keys = (
                "pixel_values", "image_grid_thw", "mm_token_type_ids", "token_type_ids",
            )

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_vig(
        self,
        cot_steps: List[str],
        question: str,
        image=None,
        image_inputs: Optional[Dict] = None,
    ) -> List[StepScore]:
        """
        高效计算每个步骤的 VIG 得分（2 次 forward）。
        在 forward 前后清理 GPU cache 避免 OOM。

        comparison_mode 决定第二次 forward 的输入：
          "no_image"     : 去掉 vision token，仅保留文本（原始行为）。
          "masked_image" : 传入随机 mask mask_ratio 比例 patch 的图像（消融 3）。
        """
        if not cot_steps:
            return []

        # 避免 gradient checkpointing + @torch.no_grad() 在 hidden_size ≠
        # num_heads × head_dim 的模型（如 Qwen3-VL-4B：hidden_size=2560，
        # 但 num_heads×head_dim=4096）下产生 0-dim attention 输出的问题。
        # modeling_layers.py:92 只在 self.training=True 时调用
        # _gradient_checkpointing_func；切换 eval 模式可安全绕开该路径，
        # forward 结束后在 finally 块中恢复原始训练状态。
        _was_training = self.model.training
        if _was_training:
            self.model.eval()

        try:
            # 拼接所有步骤为一个完整文本
            full_cot_text = "\n".join(cot_steps)

            # --- 编码各步骤，记录 token 边界 ---
            step_token_ids = [
                self.tokenizer.encode(step, add_special_tokens=False)
                for step in cot_steps
            ]
            newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)

            gen_ids: List[int] = []
            step_boundaries: List[Tuple[int, int]] = []
            for i, s_ids in enumerate(step_token_ids):
                if i > 0:
                    gen_ids.extend(newline_ids)
                start = len(gen_ids)
                gen_ids.extend(s_ids)
                step_boundaries.append((start, len(gen_ids)))

            gen_len = len(gen_ids)

            # 清理 GPU cache
            torch.cuda.empty_cache()

            # --- 有图 forward ---
            if image is not None and self.processor is not None:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ]},
                    {"role": "assistant", "content": full_cot_text},
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                vis_inputs = self.processor(
                    text=[text], images=[image], return_tensors="pt"
                )
                vis_input_ids = vis_inputs["input_ids"].to(self.device)

                model_kwargs = {"input_ids": vis_input_ids}
                for k in self._mm_forward_keys:
                    if k in vis_inputs:
                        model_kwargs[k] = vis_inputs[k].to(self.device)

                outputs_vis = self.model(**model_kwargs)
                # 只取生成部分对应位置的 logits 算 entropy，立即释放完整 logits
                vis_ids_list = vis_input_ids[0].tolist()
                vis_offset = self._find_gen_offset(vis_ids_list, gen_ids)
                # 取 [vis_offset-1 : vis_offset+gen_len-1] 的 logits（预测 gen 部分）
                vis_logits_slice = outputs_vis.logits[0, max(0, vis_offset-1):vis_offset+gen_len-1].clone()
                del outputs_vis, model_kwargs
                torch.cuda.empty_cache()

                entropy_vis = self._logits_to_entropy(vis_logits_slice)
                del vis_logits_slice

                # --- 第二次 forward（comparison_mode 决定方式）---
                if self.comparison_mode == "masked_image":
                    # 消融 3：传入随机 mask 50% patch 的图像
                    masked_image = self._mask_image_random(image, mask_ratio=self.mask_ratio)
                    masked_vis_inputs = self.processor(
                        text=[text], images=[masked_image], return_tensors="pt"
                    )
                    masked_input_ids = masked_vis_inputs["input_ids"].to(self.device)
                    model_kwargs2 = {"input_ids": masked_input_ids}
                    for k in self._mm_forward_keys:
                        if k in masked_vis_inputs:
                            model_kwargs2[k] = masked_vis_inputs[k].to(self.device)
                    outputs_masked = self.model(**model_kwargs2)
                    masked_ids_list = masked_input_ids[0].tolist()
                    masked_offset = self._find_gen_offset(masked_ids_list, gen_ids)
                    masked_logits_slice = outputs_masked.logits[
                        0,
                        max(0, masked_offset - 1):masked_offset + gen_len - 1
                    ].clone()
                    del outputs_masked, model_kwargs2, masked_vis_inputs, masked_input_ids
                    torch.cuda.empty_cache()
                    entropy_txt = self._logits_to_entropy(masked_logits_slice)
                    del masked_logits_slice
                else:
                    # 原始行为（"no_image"）：去掉 vision token → 纯文本 forward
                    txt_ids_list = self._remove_vision_tokens(vis_ids_list)
                    del vis_inputs, vis_input_ids
                    torch.cuda.empty_cache()

                    txt_input_ids = torch.tensor(
                        [txt_ids_list], dtype=torch.long, device=self.device
                    )
                    outputs_txt = self.model(input_ids=txt_input_ids)
                    txt_offset = self._find_gen_offset(txt_ids_list, gen_ids)
                    txt_logits_slice = outputs_txt.logits[0, max(0, txt_offset-1):txt_offset+gen_len-1].clone()
                    del outputs_txt, txt_input_ids
                    torch.cuda.empty_cache()

                    entropy_txt = self._logits_to_entropy(txt_logits_slice)
                    del txt_logits_slice

            else:
                # 无图像：一次 forward，VIG=0
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": full_cot_text},
                ]
                if self.processor:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    txt_inputs = self.processor(text=[text], return_tensors="pt")
                    txt_input_ids = txt_inputs["input_ids"].to(self.device)
                else:
                    ids = self.tokenizer.encode(
                        f"{question}\n{full_cot_text}", return_tensors="pt"
                    )
                    txt_input_ids = ids.to(self.device)

                outputs_txt = self.model(input_ids=txt_input_ids)
                txt_ids_list = txt_input_ids[0].tolist()
                txt_offset = self._find_gen_offset(txt_ids_list, gen_ids)
                txt_logits_slice = outputs_txt.logits[0, max(0, txt_offset-1):txt_offset+gen_len-1].clone()
                del outputs_txt, txt_input_ids
                torch.cuda.empty_cache()

                entropy_txt = self._logits_to_entropy(txt_logits_slice)
                entropy_vis = entropy_txt
                del txt_logits_slice

            # --- 按步骤边界计算 ---
            results: List[StepScore] = []
            for idx, (step, (start, end)) in enumerate(zip(cot_steps, step_boundaries)):
                n_tokens = end - start
                if n_tokens == 0:
                    results.append(StepScore(
                        step=step, index=idx,
                        H_vis=0.0, H_txt=0.0, VIG=0.0,
                    ))
                    continue

                # entropy 现在只包含 gen 部分（长度 = gen_len）
                # step 在 gen_ids 中的位置 [start, end)
                # 对应 entropy 位置 [start, end)（因为我们已截取对应 slice）
                v_end = min(end, len(entropy_vis))
                t_end = min(end, len(entropy_txt))

                if v_end <= start or t_end <= start:
                    results.append(StepScore(
                        step=step, index=idx,
                        H_vis=0.0, H_txt=0.0, VIG=0.0,
                    ))
                    continue

                vis_seg = entropy_vis[start:v_end]
                txt_seg = entropy_txt[start:t_end]

                if self.split_high_variance:
                    # 高方差 step 切分为子段，每个子段独立计算 StepScore
                    sub_bounds = self._split_by_variance(vis_seg, self.variance_threshold)
                    for sub_start, sub_end in sub_bounds:
                        vs = vis_seg[sub_start:sub_end]
                        ts = txt_seg[sub_start:sub_end] if sub_end <= len(txt_seg) else txt_seg[sub_start:]
                        H_vis_sub = self._aggregate_entropy(vs)
                        H_txt_sub = self._aggregate_entropy(ts)
                        results.append(StepScore(
                            step=step, index=idx,
                            H_vis=H_vis_sub,
                            H_txt=H_txt_sub,
                            VIG=H_txt_sub - H_vis_sub,
                            H_vis_std=vis_seg[sub_start:sub_end].std().item() if len(vs) > 1 else 0.0,
                            H_vis_max=vis_seg[sub_start:sub_end].max().item(),
                        ))
                else:
                    H_vis = self._aggregate_entropy(vis_seg)
                    H_txt = self._aggregate_entropy(txt_seg)
                    results.append(StepScore(
                        step=step, index=idx,
                        H_vis=H_vis, H_txt=H_txt,
                        VIG=H_txt - H_vis,
                        H_vis_std=vis_seg.std().item() if len(vis_seg) > 1 else 0.0,
                        H_vis_max=vis_seg.max().item(),
                    ))

            return results

        finally:
            if _was_training:
                self.model.train()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _aggregate_entropy(self, h: torch.Tensor) -> float:
        """
        将 step 内 token 级别的熵序列聚合为标量。

        aggregation_mode 参数控制聚合策略：
          "mean"     — 原始均值（默认行为，与旧版本一致）。
          "max_mean" — 0.5*max + 0.5*mean（兼顾均值和峰值，避免
                       关键高熵 token 被大量低熵结构词稀释）。
          "max_std"  — 0.5*mean + 0.3*max + 0.2*std（在 max_mean
                       基础上加入方差，奖励步内熵分布不均匀的 step）。
        """
        if len(h) == 0:
            return 0.0
        mean_h = h.mean().item()
        if self.aggregation_mode == "mean":
            return mean_h
        max_h = h.max().item()
        if self.aggregation_mode == "max_mean":
            return 0.5 * mean_h + 0.5 * max_h
        # "max_std"
        std_h = h.std().item() if len(h) > 1 else 0.0
        return 0.5 * mean_h + 0.3 * max_h + 0.2 * std_h

    def _split_by_variance(
        self, h: torch.Tensor, threshold: float
    ) -> List[Tuple[int, int]]:
        """
        将高方差 step 的熵序列按高/低熵分界切分为子段。

        策略：
          1. 若 std(h) < threshold，整段不切分（返回单个子段）。
          2. 否则以中位数为阈值，扫描状态转换点，将连续高熵 token
             和连续低熵 token 各归为一个子段。

        返回 [(sub_start, sub_end), ...] 列表，覆盖 [0, len(h))。
        """
        if len(h) == 0:
            return [(0, 0)]
        if len(h) == 1 or (len(h) > 1 and h.std().item() < threshold):
            return [(0, len(h))]

        median_h = h.median().item()
        boundaries = [0]
        in_high = (h[0].item() > median_h)
        for i in range(1, len(h)):
            cur_high = (h[i].item() > median_h)
            if cur_high != in_high:
                boundaries.append(i)
                in_high = cur_high
        boundaries.append(len(h))
        return list(zip(boundaries[:-1], boundaries[1:]))

    def _logits_to_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits [seq_len, vocab] → Shannon 熵 [seq_len]
        分 chunk 计算以节省显存（避免一次性 softmax 整个 vocab 维度）。
        """
        seq_len = logits.shape[0]
        entropy = torch.zeros(seq_len, device=logits.device)
        chunk_size = 128  # 每次处理 128 个 token 位置

        for i in range(0, seq_len, chunk_size):
            chunk = logits[i:i + chunk_size]  # [chunk, vocab]
            log_p = F.log_softmax(chunk, dim=-1)
            p = torch.exp(log_p)
            entropy[i:i + chunk_size] = -torch.sum(p * log_p, dim=-1)

        return entropy

    def _remove_vision_tokens(self, token_ids: List[int]) -> List[int]:
        """
        删除 input_ids 中的图像 token 区段（跨架构）。

        marker 模式（qwen3vl / internvl / gemma3）：
          删掉 start_marker 到 end_marker（含两端）之间所有 token。
          qwen3vl: <|vision_start|>...<|vision_end|>（原始行为，逐 token 语义不变）
          internvl: <img><IMG_CONTEXT>*N</img>
          gemma3:   <start_of_image><image_soft_token>*N<end_of_image>
        auto 模式：
          删除所有等于 image_token_id 的 token（等价于删除其连续段）。
        """
        if self.vision_span_mode == "auto":
            return [tid for tid in token_ids if tid != self.image_token_id]

        result = []
        skip = False
        for tid in token_ids:
            if tid == self.vision_start_id:
                skip = True
                continue
            if tid == self.vision_end_id:
                skip = False
                continue
            if not skip:
                result.append(tid)
        return result

    def _mask_image_random(self, image, mask_ratio: float = 0.5):
        """
        随机 mask 图像中 mask_ratio 比例的 patch（以 14x14 为单位），用灰色填充。

        适用于 PIL Image（RGB）。若图像不是 PIL Image，直接返回原图（fallback）。
        mask_ratio=0.5 → 随机遮挡约 50% 的 patch 面积。
        """
        try:
            import numpy as np
            from PIL import Image as PILImage

            if not hasattr(image, "mode"):
                return image
            img = image.convert("RGB")
            arr = np.array(img, dtype=np.float32)  # [H, W, 3]
            H, W, _ = arr.shape

            patch_size = 14
            n_ph = max(H // patch_size, 1)
            n_pw = max(W // patch_size, 1)
            n_patches = n_ph * n_pw

            # 随机选取 mask_ratio 比例的 patch
            n_mask = max(1, int(round(n_patches * mask_ratio)))
            mask_indices = np.random.choice(n_patches, size=n_mask, replace=False)

            # 计算灰色填充值（0.5 * 255 = 127.5）
            fill_value = 127.5

            for flat_idx in mask_indices:
                row_idx = flat_idx // n_pw
                col_idx = flat_idx % n_pw
                r0 = row_idx * patch_size
                r1 = min(r0 + patch_size, H)
                c0 = col_idx * patch_size
                c1 = min(c0 + patch_size, W)
                arr[r0:r1, c0:c1, :] = fill_value

            masked_img = PILImage.fromarray(arr.astype(np.uint8))
            return masked_img

        except Exception:
            # fallback：返回原图
            return image

    def _find_gen_offset(self, full_ids: List[int], gen_ids: List[int]) -> int:
        """
        在 full_ids 中找到 gen_ids 子序列的起始位置。
        从末尾向前搜索（生成文本通常在序列末尾）。
        """
        gen_len = len(gen_ids)
        if gen_len == 0:
            return len(full_ids)

        # 从末尾向前搜索
        for i in range(len(full_ids) - gen_len, -1, -1):
            if full_ids[i:i + gen_len] == gen_ids:
                return i

        # 如果精确匹配失败，尝试用前几个 token 做模糊定位
        # 取 gen_ids 前 5 个 token 做搜索
        prefix = gen_ids[:min(5, gen_len)]
        prefix_len = len(prefix)
        for i in range(len(full_ids) - prefix_len, -1, -1):
            if full_ids[i:i + prefix_len] == prefix:
                return i

        # fallback：假设生成文本在末尾
        return len(full_ids) - gen_len

    # ------------------------------------------------------------------
    # 批量处理
    # ------------------------------------------------------------------

    def batch_compute_vig(
        self,
        samples: List[Dict],
        image_key: str = "image",
        question_key: str = "question",
        cot_key: str = "cot_steps",
    ) -> List[List[StepScore]]:
        """批量计算 VIG（逐条处理，每条 2 次 forward）。"""
        all_scores = []
        for sample in samples:
            scores = self.compute_vig(
                cot_steps=sample[cot_key],
                question=sample[question_key],
                image=sample.get(image_key),
            )
            all_scores.append(scores)
        return all_scores
