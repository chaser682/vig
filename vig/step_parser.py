"""
StepParser: 将 CoT 文本按步骤分割。

支持多种分割模式（step_split_mode）：
  "newline"   : 默认。优先按编号标记，退而按换行，最后按句末标点。
  "sentence"  : 消融：仅按句末标点（. / 。/ ？/ ！等）分割，不依赖换行和编号。
  "token"     : 消融：不做步骤切分，整个 <think> 块视为一个"步骤"，
                对应 compute_vig 层面的 token 级别直接均值（无步骤粒度）。
"""

import re
from typing import List


class StepParser:
    """将 Chain-of-Thought 文本解析成步骤列表。"""

    # --- 内置分割规则（优先级从高到低）---
    NUMBERED_PATTERN = re.compile(
        r"(?:^|\n)"
        r"(?:"
        r"(?:步骤|Step|STEP)\s*\d+[\.、:：]?"  # "Step 1:", "步骤1."
        r"|(?:\(\d+\)|\[\d+\])"               # "(1)", "[2]"
        r"|(?:\d+[\.、]\s)"                    # "1. ", "2、"
        r")"
        r"(.+?)(?=(?:\n(?:步骤|Step|STEP)\s*\d+|\n\(\d+\)|\n\[\d+\]|\n\d+[\.、])|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    NEWLINE_PATTERN = re.compile(r"\n{1,}")

    # 句末标点分割：中英文句号/感叹号/问号（sentence 模式专用）
    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？.!?])\s*")

    def __init__(self, min_step_tokens: int = 5, step_split_mode: str = "newline"):
        """
        Args:
            min_step_tokens: 低于该词数的片段将被过滤（避免空行/单词片段）。
            step_split_mode: 分割模式，见模块文档说明。
                "newline"  — 默认：编号 > 换行 > 句末标点（原始行为）。
                "sentence" — 消融：仅按句末标点（. / 。）分割。
                "token"    — 消融：不切分，整个 think 块作为一个步骤，
                             实现 token 级别的 avg_VIG。
        """
        self.min_step_tokens = min_step_tokens
        self.step_split_mode = step_split_mode

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def parse(self, cot_text: str) -> List[str]:
        """
        按 step_split_mode 解析步骤。

        Returns:
            steps: 去除前后空白的步骤字符串列表。
        """
        cot_text = cot_text.strip()
        if not cot_text:
            return []

        if self.step_split_mode == "token":
            # token 级别模式：整块作为单一步骤，VIGCompressor 按 token 均值计算
            return [cot_text]

        if self.step_split_mode == "sentence":
            # 仅按句末标点分割（. / 。/ ？/ ！），不走编号/换行逻辑
            steps = self._parse_by_sentence(cot_text)
            return steps if steps else [cot_text]

        # 默认 "newline" 模式：原始行为
        # 优先尝试带编号的解析
        steps = self._parse_numbered(cot_text)
        if len(steps) >= 2:
            return steps

        # 退而按换行分割
        steps = self._parse_by_newline(cot_text)
        if len(steps) >= 2:
            return steps

        # 最后按句号/问号/感叹号分割
        steps = self._parse_by_sentence(cot_text)
        return steps if steps else [cot_text]

    def parse_think_block(self, full_output: str) -> List[str]:
        """
        从 <think>...</think> 块中提取并解析步骤。

        若无 <think> 标签，则对整体文本解析。
        """
        think_match = re.search(r"<think>(.*?)</think>", full_output, re.DOTALL)
        if think_match:
            cot_text = think_match.group(1).strip()
        else:
            cot_text = full_output.strip()
        return self.parse(cot_text)

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _parse_numbered(self, text: str) -> List[str]:
        matches = self.NUMBERED_PATTERN.findall(text)
        if not matches:
            # fallback: 尝试按 "\n数字." 分割
            parts = re.split(r"\n\d+[\.、]\s+", text)
            if len(parts) >= 2:
                return [p.strip() for p in parts if self._valid(p)]
            return []
        return [m.strip() for m in matches if self._valid(m)]

    def _parse_by_newline(self, text: str) -> List[str]:
        parts = self.NEWLINE_PATTERN.split(text)
        return [p.strip() for p in parts if self._valid(p)]

    def _parse_by_sentence(self, text: str) -> List[str]:
        # 按中英文句末标点分割，保留标点
        parts = self.SENTENCE_SPLIT_PATTERN.split(text)
        return [p.strip() for p in parts if self._valid(p)]

    def _valid(self, s: str) -> bool:
        """步骤有效性检查：非空 + 词数 ≥ min_step_tokens。"""
        s = s.strip()
        if not s:
            return False
        # 简单 token 估计（按空白 + 中文字符）
        token_count = len(s.split()) + len(re.findall(r"[一-鿿]", s))
        return token_count >= self.min_step_tokens


# ------------------------------------------------------------------
# 简单测试
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
Step 1: From the image, the height of pole A is h1 = 80 cm.
Step 2: The shadow length of pole A is s1 = 120 cm.
Step 3: We need to carefully analyze this problem to find the answer.
Step 4: By similar triangles, h1/s1 = h2/s2, so h2 = h1 * s2 / s1 = 80 * 150 / 120 = 100 cm.
Step 5: Therefore, the height of pole B is 100 cm.
"""
    for mode in ("newline", "sentence", "token"):
        parser = StepParser(step_split_mode=mode)
        steps = parser.parse(sample)
        print(f"\n[mode={mode}] {len(steps)} steps:")
        for i, s in enumerate(steps):
            print(f"  [{i}] {s[:80]}")
