"""
RL 训练数据集加载模块。

支持数据集（来自 idea.txt 中的 RL 数据集）：
  - Lin-Chen/MMStar         通用视觉理解，val，~1500 条
  - AI4Math/MathVista        视觉数学推理，testmini，~1000 条
  - lscpku/LogicVista        视觉逻辑推理，test，~448 条

统一输出格式：
    {
        "id":            str,
        "question":      str,         # 问题文本
        "image":         PIL.Image or None,
        "answer":        str,         # ground truth 答案
        "options":       List[str] or None,  # 选择题选项（若有）
        "source":        str,         # 数据集来源标识
        "original_cot":  str or None, # 原始 CoT（若有）
    }
"""

import os
import random
from typing import List, Dict, Optional, Iterator

from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _format_options(options: List[str]) -> str:
    """将选项列表格式化为 (A) xxx (B) xxx ... 字符串。"""
    labels = "ABCDEFGH"
    return " ".join(f"({labels[i]}) {opt}" for i, opt in enumerate(options))


def _build_question_text(question: str, options: Optional[List[str]] = None) -> str:
    """将问题 + 选项拼接成完整提问文本。"""
    if options:
        return f"{question}\nOptions: {_format_options(options)}"
    return question


# ---------------------------------------------------------------------------
# MMStar 数据集
# ---------------------------------------------------------------------------

class MMStarDataset(Dataset):
    """
    Lin-Chen/MMStar (val split)。
    HuggingFace 格式：含 image, question, answer 字段。
    """

    HF_DATASET_ID = "Lin-Chen/MMStar"
    SPLIT = "val"
    SOURCE = "mmstar"

    def __init__(self, cache_dir: Optional[str] = None, max_samples: Optional[int] = None):
        from datasets import load_dataset
        ds = load_dataset(self.HF_DATASET_ID, split=self.SPLIT, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        image = item.get("image", None)
        question = item.get("question", "")
        answer = str(item.get("answer", ""))
        options = item.get("options", None)

        return {
            "id": f"mmstar_{idx}",
            "question": _build_question_text(question, options),
            "image": image,
            "answer": answer,
            "options": options,
            "source": self.SOURCE,
            "original_cot": None,
        }


# ---------------------------------------------------------------------------
# MathVista 数据集
# ---------------------------------------------------------------------------

class MathVistaDataset(Dataset):
    """
    AI4Math/MathVista (testmini split)。
    """

    HF_DATASET_ID = "AI4Math/MathVista"
    SPLIT = "testmini"
    SOURCE = "mathvista"

    def __init__(self, cache_dir: Optional[str] = None, max_samples: Optional[int] = None):
        from datasets import load_dataset
        ds = load_dataset(self.HF_DATASET_ID, split=self.SPLIT, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        image = item.get("image", None)
        question = item.get("question", "")
        answer = str(item.get("answer", ""))
        options = item.get("choices", item.get("options", None))

        return {
            "id": f"mathvista_{idx}",
            "question": _build_question_text(question, options),
            "image": image,
            "answer": answer,
            "options": options,
            "source": self.SOURCE,
            "original_cot": None,
        }


# ---------------------------------------------------------------------------
# LogicVista 数据集
# ---------------------------------------------------------------------------

class LogicVistaDataset(Dataset):
    """
    lscpku/LogicVista (test split)。
    """

    HF_DATASET_ID = "lscpku/LogicVista"
    SPLIT = "test"
    SOURCE = "logicvista"

    def __init__(self, cache_dir: Optional[str] = None, max_samples: Optional[int] = None):
        from datasets import load_dataset
        ds = load_dataset(self.HF_DATASET_ID, split=self.SPLIT, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        # LogicVista 字段适配（字段名可能略有差异）
        image = item.get("image", None)
        question = item.get("question", item.get("problem", ""))
        answer = str(item.get("answer", item.get("label", "")))
        options = item.get("choices", item.get("options", None))

        return {
            "id": f"logicvista_{idx}",
            "question": _build_question_text(question, options),
            "image": image,
            "answer": answer,
            "options": options,
            "source": self.SOURCE,
            "original_cot": None,
        }


# ---------------------------------------------------------------------------
# 混合数据集（用于 RL 训练）
# ---------------------------------------------------------------------------

class RLMixedDataset(Dataset):
    """
    将 MMStar + MathVista + LogicVista 合并为统一 RL 训练集。

    Args:
        cache_dir:       HuggingFace 数据集缓存路径。
        max_per_source:  每个数据源的最大样本数（None 表示不限）。
        shuffle:         是否打乱混合后的数据集。
        seed:            随机种子。
        sources:         指定使用哪些数据源，默认全部。
    """

    REGISTRY = {
        "mmstar":     MMStarDataset,
        "mathvista":  MathVistaDataset,
        "logicvista": LogicVistaDataset,
    }

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_per_source: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        sources: Optional[List[str]] = None,
    ):
        if sources is None:
            sources = list(self.REGISTRY.keys())

        self.samples: List[Dict] = []
        for src in sources:
            cls = self.REGISTRY[src]
            ds = cls(cache_dir=cache_dir, max_samples=max_per_source)
            print(f"[RLMixedDataset] Loaded {len(ds)} samples from {src}")
            for i in range(len(ds)):
                self.samples.append(ds[i])

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.samples)

        print(f"[RLMixedDataset] Total {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collate 函数（用于 DataLoader）
# ---------------------------------------------------------------------------

def rl_collate_fn(batch: List[Dict]) -> Dict:
    """
    将 List[Dict] 组成 batch dict。
    image 保持为 list（每条可能是不同尺寸的 PIL.Image，不做 tensor 化）。
    """
    keys = batch[0].keys()
    collated = {k: [item[k] for item in batch] for k in keys}
    return collated


def build_rl_dataloader(
    cache_dir: Optional[str] = None,
    batch_size: int = 4,
    max_per_source: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 2,
    sources: Optional[List[str]] = None,
) -> DataLoader:
    """
    构建 RL 训练用 DataLoader。
    """
    dataset = RLMixedDataset(
        cache_dir=cache_dir,
        max_per_source=max_per_source,
        shuffle=shuffle,
        sources=sources,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 已在 Dataset 内部 shuffle
        num_workers=num_workers,
        collate_fn=rl_collate_fn,
    )
    return loader


# ---------------------------------------------------------------------------
# 系统提示
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. "
    "Think step by step, focusing on information directly relevant to solving the problem. "
    "Avoid redundant or generic statements. "
    "Format your response as:\n"
    "<think>\n[Your concise step-by-step reasoning]\n</think>\n"
    "<answer>\n[Your final answer]\n</answer>"
)


def build_prompt(question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """构建完整的用户 prompt（不含图像，图像通过 processor 单独处理）。"""
    return f"{system_prompt}\n\nQuestion: {question}"
