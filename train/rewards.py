"""
奖励函数模块：GRPO 训练用。

奖励组成（不含 R_len）：
    R_total = R_format + R_acc + R_VIG

各项说明
--------
R_format:  检查输出是否符合 <think>...</think><answer>...</answer> 格式
R_acc:     答案正确性（精确匹配 / 归一化匹配）
R_VIG:    视觉信息增益奖励（avg_VIG，鼓励每步都有视觉价值）
"""

import re
import string
from typing import List, Dict, Optional, Tuple

from vig.compute_vig import VIGCompressor
from vig.step_parser import StepParser


# ---------------------------------------------------------------------------
# 格式奖励
# ---------------------------------------------------------------------------

def compute_format_reward(output: str) -> float:
    """
    检查模型输出是否包含 <think>...</think> 和 <answer>...</answer>。

    Returns:
        1.0  完整格式
        0.5  只有 <answer> 标签
        0.0  无任何规范标签
    """
    has_think = bool(re.search(r"<think>.*?</think>", output, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", output, re.DOTALL))

    if has_think and has_answer:
        return 1.0
    elif has_answer:
        return 0.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# 答案提取工具
# ---------------------------------------------------------------------------

def extract_answer(output: str) -> str:
    """
    从模型输出中提取答案字符串。

    优先级：
    1. <answer>...</answer> 标签内容
    2. </think> 之后的文本（取最后一个非空行）
    3. 整个输出的最后一个非空行
    """
    # 1. <answer> 标签
    match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. </think> 之后的文本
    if "</think>" in output:
        after_think = output.split("</think>")[-1].strip()
        if after_think:
            # 取最后一个非空行
            lines = [l.strip() for l in after_think.split("\n") if l.strip()]
            if lines:
                return lines[-1]

    # 3. fallback: 最后一个非空行
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _extract_mc_letter(text: str) -> str:
    """从文本中提取 MC 选项字母 (A-H)。"""
    text = text.strip()
    # 单字母
    if len(text) == 1 and text.upper() in "ABCDEFGH":
        return text.upper()
    # (A) 或 A 格式
    clean = text.strip("().").strip()
    if len(clean) == 1 and clean.upper() in "ABCDEFGH":
        return clean.upper()
    # "The answer is A" / "Answer: B"
    match = re.search(r'(?:answer\s*(?:is|:)\s*)\(?([A-H])\)?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 短文本中的独立字母
    if len(text) < 20:
        match = re.search(r'\b([A-H])\b', text)
        if match:
            return match.group(1).upper()
    return ""


def normalize_answer(s: str) -> str:
    """统一大小写、去标点、去多余空格，用于宽松匹配。"""
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_numeric(s: str) -> Optional[float]:
    """尝试从字符串中提取数字（支持分数、小数）。"""
    # 分数
    frac = re.search(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if frac:
        try:
            return int(frac.group(1)) / int(frac.group(2))
        except ZeroDivisionError:
            pass
    # 普通数字（含负数、小数、科学计数）
    num = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
    if num:
        try:
            return float(num.group())
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# 答案正确性奖励
# ---------------------------------------------------------------------------

def compute_accuracy_reward(
    output: str,
    ground_truth: str,
    choices: Optional[List[str]] = None,
    numeric_tolerance: float = 1e-3,
) -> float:
    """
    计算答案正确性奖励（支持 MC 字母↔选项文本双向匹配）。

    匹配策略（优先级从高到低）：
      1. 精确匹配（大小写归一化）
      2. MC 选项匹配：pred 是字母 → 映射为选项文本与 gold 比较
      3. MC 反向匹配：gold 是字母 → 映射为选项文本与 pred 比较
      4. 数值匹配（支持容差）
      5. 包含匹配（ground_truth 是 pred 的子串）

    Args:
        output:           模型完整输出文本
        ground_truth:     参考答案
        choices:          选项列表 (如 ["135°", "140°", "145°", "150°"])
        numeric_tolerance: 数值比较容差

    Returns:
        1.0  正确
        0.5  包含匹配
        0.0  错误
    """
    pred = extract_answer(output)
    gold = ground_truth.strip()

    if not pred:
        return 0.0

    # --- 精确匹配（归一化后）---
    if normalize_answer(pred) == normalize_answer(gold):
        return 1.0

    # --- MC 选项匹配（choices 不为空时启用）---
    if choices:
        labels = "ABCDEFGH"

        # 从 pred 中提取字母
        pred_letter = _extract_mc_letter(pred)

        # 情况 1: pred 是字母，gold 是选项文本
        # 例如 pred="C", gold="145°", choices=["135°","140°","145°","150°"]
        if pred_letter:
            idx = ord(pred_letter) - ord('A')
            if idx < len(choices):
                pred_value = choices[idx].strip()
                if normalize_answer(pred_value) == normalize_answer(gold):
                    return 1.0
                # 也尝试数值比较
                pv_num = extract_numeric(pred_value)
                g_num = extract_numeric(gold)
                if pv_num is not None and g_num is not None:
                    if abs(pv_num - g_num) <= numeric_tolerance * (abs(g_num) + 1e-8):
                        return 1.0

        # 情况 2: gold 是字母，pred 是选项文本
        # 例如 gold="C", pred="145°", choices=["135°","140°","145°","150°"]
        gold_letter = _extract_mc_letter(gold)
        if gold_letter:
            gold_idx = ord(gold_letter) - ord('A')
            if gold_idx < len(choices):
                gold_value = choices[gold_idx].strip()
                if normalize_answer(pred) == normalize_answer(gold_value):
                    return 1.0
                pn = extract_numeric(pred)
                gvn = extract_numeric(gold_value)
                if pn is not None and gvn is not None:
                    if abs(pn - gvn) <= numeric_tolerance * (abs(gvn) + 1e-8):
                        return 1.0

        # 情况 3: pred 和 gold 都是字母
        if pred_letter and gold_letter and pred_letter == gold_letter:
            return 1.0

        # 情况 4: pred 是选项文本，检查它对应哪个字母
        for i, choice_text in enumerate(choices):
            if normalize_answer(pred) == normalize_answer(choice_text):
                # pred 匹配到了选项 i，检查 gold 是否也是选项 i
                if gold_letter and (ord(gold_letter) - ord('A')) == i:
                    return 1.0
                if normalize_answer(gold) == normalize_answer(choice_text):
                    return 1.0
                break

    # --- 数值匹配 ---
    pred_num = extract_numeric(pred)
    gold_num = extract_numeric(gold)
    if pred_num is not None and gold_num is not None:
        if abs(pred_num - gold_num) <= numeric_tolerance * (abs(gold_num) + 1e-8):
            return 1.0

    # --- 宽松包含匹配 ---
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if gold_norm and gold_norm in pred_norm:
        return 0.5

    return 0.0


# ---------------------------------------------------------------------------
# VIG 奖励
# ---------------------------------------------------------------------------

def compute_vig_reward(
    generated_output: str,
    original_cot_steps: List[str],
    question: str,
    vig_compressor: VIGCompressor,
    image=None,
    image_inputs: Optional[Dict] = None,
    use_simple: bool = False,
    scale: float = 1.0,
    clip_range: Tuple[float, float] = (-2.0, 2.0),
) -> float:
    """
    计算 R_VIG（视觉感知步骤熵奖励）。

    调用 VIGCompressor.compute_vig_reward() 并做 scale + clip 后处理。

    Args:
        generated_output:   模型完整生成文本（含 <think> 标签）。
        original_cot_steps: 原始参考 CoT 步骤列表。
        question:           问题文本。
        vig_compressor:    已初始化的 VIGCompressor 实例。
        image/image_inputs: 图像输入。
        use_simple:         是否使用简化版 R_VIG（只用平均 VIG）。
        scale:              对原始奖励值的缩放系数。
        clip_range:         对奖励值的截断范围，避免极端值破坏训练。

    Returns:
        float: 处理后的 R_VIG 值。
    """
    raw_reward = vig_compressor.compute_vig_reward(
        generated_output=generated_output,
        original_cot_steps=original_cot_steps,
        question=question,
        image=image,
        image_inputs=image_inputs,
        use_simple=use_simple,
    )
    scaled = raw_reward * scale
    clipped = max(clip_range[0], min(clip_range[1], scaled))
    return clipped


# ---------------------------------------------------------------------------
# 总奖励（无 R_len）
# ---------------------------------------------------------------------------

def compute_total_reward(
    output: str,
    ground_truth: str,
    original_cot_steps: List[str],
    question: str,
    vig_compressor: VIGCompressor,
    image=None,
    image_inputs: Optional[Dict] = None,
    # 权重
    w_format: float = 0.1,
    w_acc: float = 1.0,
    w_vig: float = 0.5,
    # VIG 奖励配置
    use_simple_vig: bool = False,
    vig_scale: float = 1.0,
    vig_clip: Tuple[float, float] = (-2.0, 2.0),
) -> Dict[str, float]:
    """
    计算并返回所有奖励分量及总奖励。

    奖励公式：
        R_total = w_format * R_format + w_acc * R_acc + w_vig * R_VIG

    Returns:
        dict 包含：
          - R_format, R_acc, R_VIG
          - R_total
    """
    r_format = compute_format_reward(output)
    r_acc = compute_accuracy_reward(output, ground_truth)
    r_vig = compute_vig_reward(
        generated_output=output,
        original_cot_steps=original_cot_steps,
        question=question,
        vig_compressor=vig_compressor,
        image=image,
        image_inputs=image_inputs,
        use_simple=use_simple_vig,
        scale=vig_scale,
        clip_range=vig_clip,
    )

    r_total = w_format * r_format + w_acc * r_acc + w_vig * r_vig

    return {
        "R_format": r_format,
        "R_acc": r_acc,
        "R_VIG": r_vig,
        "R_total": r_total,
    }


# ---------------------------------------------------------------------------
# 批量奖励计算（供 GRPO trainer 调用）
# ---------------------------------------------------------------------------

def batch_compute_rewards(
    outputs: List[str],
    ground_truths: List[str],
    original_cot_steps_list: List[List[str]],
    questions: List[str],
    vig_compressor: VIGCompressor,
    images: Optional[List] = None,
    image_inputs_list: Optional[List[Dict]] = None,
    **reward_kwargs,
) -> List[Dict[str, float]]:
    """
    批量计算奖励（逐条处理）。

    Args:
        outputs:                 模型生成文本列表
        ground_truths:           参考答案列表
        original_cot_steps_list: 原始 CoT 步骤列表（每条样本）
        questions:               问题列表
        vig_compressor:         VIGCompressor 实例
        images:                  图像列表（可选）
        image_inputs_list:       预处理图像 dict 列表（可选）
        **reward_kwargs:         传入 compute_total_reward 的其他参数

    Returns:
        List[Dict[str, float]]: 每条样本的奖励 dict
    """
    n = len(outputs)
    assert n == len(ground_truths) == len(questions) == len(original_cot_steps_list)

    if images is None:
        images = [None] * n
    if image_inputs_list is None:
        image_inputs_list = [None] * n

    results = []
    for i in range(n):
        reward_dict = compute_total_reward(
            output=outputs[i],
            ground_truth=ground_truths[i],
            original_cot_steps=original_cot_steps_list[i],
            question=questions[i],
            vig_compressor=vig_compressor,
            image=images[i],
            image_inputs=image_inputs_list[i],
            **reward_kwargs,
        )
        results.append(reward_dict)
    return results


# ---------------------------------------------------------------------------
# GRPO 兼容接口（直接传入 GRPOTrainer.reward_funcs）
# ---------------------------------------------------------------------------


def grpo_format_reward(prompts, completions, **kwargs) -> List[float]:
    """
    GRPO 兼容格式奖励函数。

    检查模型输出是否包含 <think>...</think> 和 <answer>...</answer> 标签。

    Returns:
         0.0  格式完整（有 think + answer）
        -0.5  只有 <answer>，缺 <think>
        -1.0  无任何规范标签
    """
    rewards = []
    for completion in completions:
        text = completion[-1]["content"] if isinstance(completion, list) else completion
        has_think = bool(re.search(r"</think>", text, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
        if has_think and has_answer:
            rewards.append(0.0)
        elif has_answer:
            rewards.append(-0.5)
        else:
            rewards.append(-1.0)
    return rewards


def grpo_accuracy_reward(prompts, completions, **kwargs) -> List[float]:
    """
    GRPO 兼容答案正确性奖励函数。

    kwargs 须含 "answer" 字段（参考答案列表，与 completions 等长）。
    可选含 "choices" 字段（选项列表，用于 MC 字母↔选项文本双向匹配）。
    复用 compute_accuracy_reward 的精确 / MC选项 / 数值 / 包含匹配策略。
    """
    ground_truths = kwargs.get("answer", [])
    choices_list = kwargs.get("choices", [])
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[-1]["content"] if isinstance(completion, list) else completion
        gold = ground_truths[i] if i < len(ground_truths) else ""
        choices = choices_list[i] if i < len(choices_list) else []
        # choices 可能是 None（某些样本无选项）
        if not choices:
            choices = None
        rewards.append(compute_accuracy_reward(text, gold, choices=choices))
    return rewards


def make_vig_reward_func(
    vig_compressor: VIGCompressor,
    lambda_: float = 1.0,
    mu: float = 0.5,
    baseline: float = 0.5,
    max_steps_per_sample: Optional[int] = None,
    score_mode: str = "full",
    gamma: float = 0.0,
    use_position_weight: bool = False,
    use_zscore_norm: bool = False,
    step_split_mode: str = "newline",
):
    """
    创建 GRPO 兼容的 VIG 奖励函数（工厂函数）。

    论文使用 score_mode="vig_only"，即 KeepScore = VIG(s_i)，对应式 (3) 的
    R_VIG = clip(mean VIG, -1, 1)。其余 score_mode 是早期探索阶段的公式变体
    （以 H_vis 加权的 "VIG" 形式），保留以便复现开发历史，论文的任何结果都
    没有使用它们；"neg_hvis" 是论文附录 F 中与视觉无关的 confidence 对照臂。

    奖励公式（原始熵值，无归一化）：
        VIG(s_i)  = H_vis(s_i) * exp(-λ * VIG(s_i))
        KeepScore(s_i) = VIG(s_i) + μ * VIG(s_i)              ← score_mode="full"（默认）
                       = VIG(s_i) + μ * VIG(s_i) + γ * H_vis * VIG ← score_mode="full_v2"（联合项）
                       = VIG(s_i)                              ← score_mode="vig_only"（消融 1）
                       = VIG(s_i)                               ← score_mode="vig_only" （消融 2）
        R_VIG = clip(avg_KeepScore - baseline, -1, 1)  ← 与原始代码一致，防止极端值破坏训练

    score_mode 参数说明：
      "full"       ：原始公式 KeepScore = H_vis*exp(-λ*VIG) + μ*VIG（默认）。
      "full_v2"    ：改进公式，在 full 基础上加 γ*H_vis*VIG 联合项，修复高 H_vis+高 VIG 被低估问题。
      "vig_only"  ：消融 1，去掉 VIG 直接补偿项，KeepScore = H_vis*exp(-λ*VIG)。
      "vig_only"   ：消融 2，只用视觉信息增益，KeepScore = VIG。

    扩展参数：
      gamma:              γ 系数（仅 score_mode="full_v2" 时生效），H_vis*VIG 联合项权重，默认 0。
      use_position_weight: 是否对 step 得分施加位置权重（倒 U 形曲线，中间步骤权重高）。
      use_zscore_norm:    是否在 GRPO group 内做 z-score 归一化（让 VIG 奖励量级稳定）。

    Args:
        vig_compressor:      已初始化的 VIGCompressor 实例。
        lambda_:              λ，VIG 对 VIG 项的指数衰减强度。
        mu:                   μ，VIG 在 KeepScore 中的加权系数。
        baseline:             从 avg_KeepScore 中减去的基准值，让奖励中心化为 0。
        max_steps_per_sample: 每条样本最多处理步骤数，None 表示不限。
        score_mode:           KeepScore 计算模式，见上方说明。
        gamma:                H_vis*VIG 联合项系数（score_mode="full_v2" 时生效）。
        use_position_weight:  是否使用 step 位置权重（中间步骤加权）。
        use_zscore_norm:      是否在 group 内 z-score 归一化奖励值。
        step_split_mode:      StepParser 分割模式：
                                "newline"  — 默认（编号 > 换行 > 句末标点）。
                                "sentence" — 消融：仅按句末标点（. / 。）分割。
                                "token"    — 消融：不切分，整 think 块为一个步骤，
                                             实现 token 级别 avg_VIG。

    Returns:
        reward_fn: GRPO 兼容签名 (prompts, completions, **kwargs) -> List[float]
    """
    import logging
    import math as _math

    _VALID_SCORE_MODES = ("full", "full_v2", "vig_only", "vig_only", "neg_hvis")
    if score_mode not in _VALID_SCORE_MODES:
        raise ValueError(f"score_mode must be one of {_VALID_SCORE_MODES}, got '{score_mode}'")

    # 使用 StepParser 进行步骤分割（step_split_mode 控制分割策略，支持消融实验）
    step_parser = StepParser(min_step_tokens=5, step_split_mode=step_split_mode)

    def _get_text(completion) -> str:
        return completion[-1]["content"] if isinstance(completion, list) else completion

    def _get_question(prompt) -> str:
        """从 prompt（字符串或对话列表）提取用户问题文本。"""
        if isinstance(prompt, list):
            for msg in prompt:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                return part.get("text", "")
                    elif isinstance(content, str):
                        return content
        return str(prompt)

    def _get_image(prompt, kwargs, idx: int):
        """从 kwargs 或 prompt 中尝试获取图像。"""
        # trl GRPOTrainer 会将 dataset 中的额外字段通过 kwargs 传递
        images = kwargs.get("image", None)
        if images is None:
            return None
        # 可能是列表也可能是单个对象
        if isinstance(images, (list, tuple)):
            if idx < len(images):
                img = images[idx]
                # 某些 dataset 中 image 字段可能为 None（文本题）
                return img if img is not None else None
        return None

    def _extract_think(text: str) -> str:
        """从完整输出中提取 <think> 块内容，无标签则返回全文（超长未闭合）。"""
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"(.*?)</think>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    def _position_weight(step_idx: int, total_steps: int) -> float:
        """
        倒 U 形位置权重：中间步骤权重最高（≈1.0），首尾步骤权重最低（≈0.5）。
        对于只有 1 步的样本，直接返回 1.0。
        """
        if total_steps <= 1:
            return 1.0
        t = step_idx / (total_steps - 1)   # 0.0 → 1.0
        return 0.5 + 0.5 * _math.sin(_math.pi * t)

    def _compute_keep_score(s, lam: float, mu_: float, gam: float) -> float:
        """根据 score_mode 计算单步 KeepScore。"""
        vig_term = s.H_vis * _math.exp(-lam * s.VIG)
        if score_mode in ("full", "full_v2"):
            score = vig_term + mu_ * s.VIG
            if score_mode == "full_v2":
                score += gam * s.H_vis * s.VIG   # 联合项：奖励高 H_vis + 高 VIG 的协同
            return score
        elif score_mode == "vig_only":
            return vig_term
        elif score_mode == "neg_hvis":
            # A1 confidence 对照臂：只奖励有图前向的低熵（自信），不含视觉对比。
            # 用于检验 VIG 的收益是否来自"视觉对比"而非一般熵塑形。
            return -s.H_vis
        else:  # "vig_only"
            return s.VIG

    def vig_reward_fn(prompts, completions, **kwargs) -> List[float]:
        """
        VIG 奖励计算（KeepScore - baseline，可选位置权重和 z-score 归一化）。

        流程：
          1. 提取 <think> 块（无标签则用全文），用 StepParser 分割为推理步骤
          2. 调用 VIGCompressor.compute_vig() 计算各步骤的 H_vis / H_txt / VIG
          3. KeepScore 由 score_mode 决定（full/full_v2/vig_only/vig_only）
          4. 可选：施加 step 位置权重（中间步骤贡献更大）
          5. R_VIG = avg_KeepScore - baseline
          6. 可选：在 group 内做 z-score 归一化（clip 到 ±2）
        """
        raw_scores = []
        valid_flags = []

        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            text = _get_text(completion)
            think_text = _extract_think(text)

            steps = step_parser.parse(think_text)

            if max_steps_per_sample is not None:
                steps = steps[:max_steps_per_sample]

            question = _get_question(prompt)
            image = _get_image(prompt, kwargs, idx)

            if not question:
                raw_scores.append(None)
                valid_flags.append(False)
                continue
            steps = [s for s in steps if s is not None]
            if not steps:
                raw_scores.append(None)
                valid_flags.append(False)
                continue

            try:
                step_scores = vig_compressor.compute_vig(
                    cot_steps=steps,
                    question=question,
                    image=image,
                    image_inputs=None,
                )
                if not step_scores:
                    raw_scores.append(None)
                    valid_flags.append(False)
                    continue

                n = len(step_scores)
                # 计算每步 KeepScore（含可选位置权重）
                weighted_sum = 0.0
                weight_total = 0.0
                for i, s in enumerate(step_scores):
                    ks = _compute_keep_score(s, lambda_, mu, gamma)
                    pw = _position_weight(i, n) if use_position_weight else 1.0
                    weighted_sum += ks * pw
                    weight_total += pw

                avg_keep_score = weighted_sum / weight_total if weight_total > 0 else 0.0
                r_vig = avg_keep_score - baseline
                r_vig = max(-1.0, min(1.0, r_vig))  # clip to (-1, 1)，与原始代码一致
                raw_scores.append(float(r_vig))
                valid_flags.append(True)

            except Exception as exc:
                import traceback
                logging.warning(f"[VIG reward] compute_vig failed: {exc}\n{traceback.format_exc()}")
                raw_scores.append(None)
                valid_flags.append(False)

        # --- z-score 归一化（仅对有效样本）---
        rewards = []
        if use_zscore_norm:
            valid_vals = [v for v in raw_scores if v is not None]
            if len(valid_vals) > 1:
                import statistics
                mean_v = statistics.mean(valid_vals)
                std_v = statistics.stdev(valid_vals) + 1e-8
            else:
                mean_v, std_v = 0.0, 1.0

            for v, flag in zip(raw_scores, valid_flags):
                if not flag or v is None:
                    rewards.append(-1.0)
                else:
                    normalized = (v - mean_v) / std_v
                    rewards.append(float(max(-2.0, min(2.0, normalized))))
        else:
            for v, flag in zip(raw_scores, valid_flags):
                if not flag or v is None:
                    rewards.append(-1.0)
                else:
                    rewards.append(v)

        return rewards

    return vig_reward_fn


# ---------------------------------------------------------------------------
# Soft Overlong Penalty（DAPO 官方长度软惩罚）
# ---------------------------------------------------------------------------

def make_soft_overlong_reward_func(
    min_len: int = 1024,
    max_len: int = 2048,
):
    """
    创建 GRPO 兼容的 soft overlong 惩罚函数（工厂函数）。

    DAPO 论文建议的长度软惩罚策略：
      - 完成长度 L ≤ min_len：无惩罚，返回 0.0
      - 完成长度 min_len < L < max_len：线性插值惩罚，从 0 → -1
      - 完成长度 L ≥ max_len：最大惩罚，返回 -1.0

    即：
        penalty(L) = 0.0                              if L ≤ min_len
                   = -(L - min_len) / (max_len - min_len)  if min_len < L < max_len
                   = -1.0                             if L ≥ max_len

    Args:
        min_len: 开始惩罚的长度阈值（含），默认 1024。
        max_len: 最大惩罚长度（含），默认 2048。

    Returns:
        reward_fn: GRPO 兼容签名 (prompts, completions, **kwargs) -> List[float]
    """
    assert min_len < max_len, f"min_len ({min_len}) must be < max_len ({max_len})"

    def _get_text(completion) -> str:
        return completion[-1]["content"] if isinstance(completion, list) else completion

    def soft_overlong_reward_fn(prompts, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            text = _get_text(completion)
            L = len(text.split())   # 按空格分词近似；也可用 token 数，这里用词数与 DAPO 论文一致
            if L <= min_len:
                rewards.append(0.0)
            elif L >= max_len:
                rewards.append(-1.0)
            else:
                rewards.append(-float(L - min_len) / float(max_len - min_len))
        return rewards

    return soft_overlong_reward_fn
