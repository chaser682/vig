"""
VIG 有效性可视化脚本
====================
验证 VIG (Visual Information Gain) 是否有效：
  - 去掉图像后，生成 token 的熵是否比带图的高？
  - 哪些 token 具有高 VIG（视觉依赖型），哪些是低 VIG（纯文本型）？

用法：
    # 快速演示（内置 hard-coded 示例，无需加载模型）
    python vig/visualize_vig.py --demo

    # 真实模型推理
    python vig/visualize_vig.py \
        --model_path Qwen/Qwen3-VL-8B-Thinking \
        --dataset mmstar \
        --sample_idx 42 \
        --output_html output/vig_visualization.html

输出：
  1. 终端彩色文本表格（token | H_vis | H_txt | VIG | bar）
  2. HTML 文件（可在浏览器中查看，按 VIG 强度着色）
  3. matplotlib 折线图（保存为 PNG）
"""

import argparse
import os
import sys
import math
import html as html_module
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F


# ── ANSI 颜色 ────────────────────────────────────────────────────────────────
def _ansi(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"

RED    = lambda s: _ansi(s, "91")
YELLOW = lambda s: _ansi(s, "93")
GREEN  = lambda s: _ansi(s, "92")
CYAN   = lambda s: _ansi(s, "96")
BOLD   = lambda s: _ansi(s, "1")
DIM    = lambda s: _ansi(s, "2")


# ── 熵计算（复用 compute_vig.py 逻辑）────────────────────────────────────────
def logits_to_entropy(logits: torch.Tensor) -> torch.Tensor:
    """logits [seq_len, vocab] → Shannon 熵 [seq_len]（chunk 计算节省显存）"""
    seq_len = logits.shape[0]
    entropy = torch.zeros(seq_len, device=logits.device)
    chunk_size = 128
    for i in range(0, seq_len, chunk_size):
        chunk = logits[i : i + chunk_size]
        log_p = F.log_softmax(chunk, dim=-1)
        p = torch.exp(log_p)
        entropy[i : i + chunk_size] = -(p * log_p).sum(dim=-1)
    return entropy


def remove_vision_tokens(token_ids: List[int], vision_start_id: int, vision_end_id: int) -> List[int]:
    """删掉 <|vision_start|>…<|vision_end|> 段（含两端）"""
    result, skip = [], False
    for tid in token_ids:
        if tid == vision_start_id:
            skip = True
            continue
        if tid == vision_end_id:
            skip = False
            continue
        if not skip:
            result.append(tid)
    return result


def find_gen_offset(full_ids: List[int], gen_ids: List[int]) -> int:
    """在 full_ids 中定位 gen_ids 子序列的起始位置（从末尾向前搜）"""
    gen_len = len(gen_ids)
    if gen_len == 0:
        return len(full_ids)
    for i in range(len(full_ids) - gen_len, -1, -1):
        if full_ids[i : i + gen_len] == gen_ids:
            return i
    # 模糊匹配前缀
    prefix = gen_ids[: min(5, gen_len)]
    for i in range(len(full_ids) - len(prefix), -1, -1):
        if full_ids[i : i + len(prefix)] == prefix:
            return i
    return len(full_ids) - gen_len


# ── 核心：token 级双向 forward ─────────────────────────────────────────────────
@torch.no_grad()
def compute_token_level_entropy(
    model,
    processor,
    tokenizer,
    image,
    question: str,
    cot_text: str,
    device: str = "cuda",
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    返回:
      tokens  : List[str]，生成 token 的文本（每个 token decode 后的字符串）
      H_vis   : Tensor [gen_len]，带图时每个 token 位置的熵
      H_txt   : Tensor [gen_len]，去图时每个 token 位置的熵
    """
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id   = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    # ── 1. 编码 gen_ids（生成文本 token 序列）──────────────────────────────────
    gen_ids: List[int] = tokenizer.encode(cot_text, add_special_tokens=False)
    gen_len = len(gen_ids)
    print(f"  生成文本 token 数: {gen_len}")

    # ── 2. 有图 forward ────────────────────────────────────────────────────────
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]},
        {"role": "assistant", "content": cot_text},
    ]
    text_template = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    vis_inputs = processor(text=[text_template], images=[image], return_tensors="pt")
    vis_input_ids = vis_inputs["input_ids"].to(device)

    model_kwargs = {"input_ids": vis_input_ids}
    for k in ("pixel_values", "image_grid_thw"):
        if k in vis_inputs:
            model_kwargs[k] = vis_inputs[k].to(device)

    print("  [1/2] 有图 forward pass...")
    outputs_vis = model(**model_kwargs)

    vis_ids_list = vis_input_ids[0].tolist()
    vis_offset = find_gen_offset(vis_ids_list, gen_ids)
    print(f"        全序列长度={len(vis_ids_list)}, gen offset={vis_offset}, gen_len={gen_len}")

    # logits[t] 预测 input_ids[t+1]，所以取 [offset-1, offset+gen_len-1]
    start_l = max(0, vis_offset - 1)
    end_l   = vis_offset + gen_len - 1
    vis_logits_slice = outputs_vis.logits[0, start_l:end_l].clone()
    del outputs_vis, model_kwargs
    torch.cuda.empty_cache()

    H_vis = logits_to_entropy(vis_logits_slice)
    del vis_logits_slice

    # ── 3. 无图 forward ────────────────────────────────────────────────────────
    txt_ids_list = remove_vision_tokens(vis_ids_list, vision_start_id, vision_end_id)
    del vis_inputs, vis_input_ids
    torch.cuda.empty_cache()

    txt_input_ids = torch.tensor([txt_ids_list], dtype=torch.long, device=device)

    print("  [2/2] 无图 forward pass...")
    outputs_txt = model(input_ids=txt_input_ids)

    txt_offset = find_gen_offset(txt_ids_list, gen_ids)
    print(f"        无图序列长度={len(txt_ids_list)}, gen offset={txt_offset}")

    start_l2 = max(0, txt_offset - 1)
    end_l2   = txt_offset + gen_len - 1
    txt_logits_slice = outputs_txt.logits[0, start_l2:end_l2].clone()
    del outputs_txt, txt_input_ids
    torch.cuda.empty_cache()

    H_txt = logits_to_entropy(txt_logits_slice)
    del txt_logits_slice

    # ── 4. decode 每个 token ────────────────────────────────────────────────────
    # 对齐 H_vis / H_txt 到 gen_ids 长度
    actual_len = min(len(H_vis), len(H_txt), gen_len)
    H_vis = H_vis[:actual_len].cpu()
    H_txt = H_txt[:actual_len].cpu()
    gen_ids_trimmed = gen_ids[:actual_len]

    tokens = [tokenizer.decode([tid]) for tid in gen_ids_trimmed]

    return tokens, H_vis, H_txt


# ── 终端可视化 ─────────────────────────────────────────────────────────────────
def print_terminal_table(
    tokens: List[str],
    H_vis: torch.Tensor,
    H_txt: torch.Tensor,
    top_k: int = 30,
    vig_threshold_high: float = 0.5,
    vig_threshold_mid: float = 0.2,
):
    VIG = H_txt - H_vis
    gen_len = len(tokens)

    print()
    print(BOLD("=" * 80))
    print(BOLD("  VIG 逐 token 分析"))
    print(BOLD("=" * 80))
    print(f"  总 token 数: {gen_len}")
    print(f"  H_vis 均值: {H_vis.mean().item():.4f} nats")
    print(f"  H_txt 均值: {H_txt.mean().item():.4f} nats")
    print(f"  VIG  均值: {VIG.mean().item():.4f} nats")
    print(f"  VIG  > 0 比例: {(VIG > 0).float().mean().item()*100:.1f}%")
    print()

    # ── 整体分布摘要 ────────────────────────────────────────────────────────────
    high_vig = (VIG > vig_threshold_high).sum().item()
    mid_vig  = ((VIG > vig_threshold_mid) & (VIG <= vig_threshold_high)).sum().item()
    low_vig  = (VIG <= vig_threshold_mid).sum().item()
    print(f"  高VIG (>{vig_threshold_high:.1f}) : {high_vig:4d} tokens  ← 强视觉依赖")
    print(f"  中VIG ({vig_threshold_mid:.1f}~{vig_threshold_high:.1f}): {mid_vig:4d} tokens  ← 中等视觉依赖")
    print(f"  低VIG (<{vig_threshold_mid:.1f}) : {low_vig:4d} tokens  ← 纯文本推理")
    print()

    # ── Top-K 高 VIG token ──────────────────────────────────────────────────────
    print(BOLD(f"  TOP-{top_k} 高 VIG Tokens（最依赖视觉的 token）:"))
    print(BOLD("  " + "-" * 76))
    print(BOLD(f"  {'Rank':>4}  {'Pos':>5}  {'Token':<20}  {'H_vis':>7}  {'H_txt':>7}  {'VIG':>7}  Bar"))
    print(BOLD("  " + "-" * 76))

    sorted_indices = VIG.argsort(descending=True)
    for rank, idx in enumerate(sorted_indices[:top_k]):
        i = idx.item()
        tok = repr(tokens[i])[:18]
        hv  = H_vis[i].item()
        ht  = H_txt[i].item()
        vig = VIG[i].item()

        bar_len = min(int(vig * 10), 20)
        bar = "█" * max(bar_len, 0)

        if vig > vig_threshold_high:
            color = RED
        elif vig > vig_threshold_mid:
            color = YELLOW
        else:
            color = GREEN

        print(f"  {rank+1:>4}  {i:>5}  {tok:<20}  {hv:>7.3f}  {ht:>7.3f}  {color(f'{vig:>7.3f}')}  {color(bar)}")

    print()

    # ── 负 VIG（无图反而更确定，模型忽略图像）──────────────────────────────────
    neg_vig_mask = VIG < -0.1
    if neg_vig_mask.sum() > 0:
        print(BOLD(f"  负 VIG Tokens（去图后模型反而更确定，这些 token 不依赖图像）:"))
        neg_indices = VIG.argsort(descending=False)[:10]
        for idx in neg_indices:
            i = idx.item()
            if VIG[i] >= -0.1:
                break
            tok = repr(tokens[i])[:18]
            print(f"    pos={i:>5}  token={tok:<20}  H_vis={H_vis[i].item():.3f}  H_txt={H_txt[i].item():.3f}  VIG={DIM(f'{VIG[i].item():.3f}')}")
        print()

    # ── 按位置展示前 100 个 token ─────────────────────────────────────────────
    print(BOLD("  前100个 token 逐步 VIG（位置顺序）:"))
    print(BOLD("  " + "-" * 76))
    print(BOLD(f"  {'Pos':>5}  {'Token':<20}  {'H_vis':>7}  {'H_txt':>7}  {'VIG':>7}  bar"))
    print(BOLD("  " + "-" * 76))
    for i in range(min(100, gen_len)):
        tok = repr(tokens[i])[:18]
        hv  = H_vis[i].item()
        ht  = H_txt[i].item()
        vig = VIG[i].item()
        bar_len = max(0, min(int(vig * 10), 15))
        bar = "▓" * bar_len if vig > 0 else "░" * min(int(-vig * 5), 5)
        if vig > vig_threshold_high:
            color = RED
        elif vig > vig_threshold_mid:
            color = YELLOW
        elif vig < -0.1:
            color = DIM
        else:
            color = GREEN
        print(f"  {i:>5}  {tok:<20}  {hv:>7.3f}  {ht:>7.3f}  {color(f'{vig:>7.3f}')}  {color(bar)}")


# ── 图像 → base64 data URI ─────────────────────────────────────────────────────
def _image_to_data_uri(image, max_size: int = 480) -> str:
    """将 PIL Image 转为 base64 data URI（限制最长边，减小文件体积）。"""
    import base64
    import io
    try:
        from PIL import Image as PILImage
        img = image.convert("RGB") if hasattr(image, "convert") else image
        # 按比例缩放
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


# ── HTML 可视化 ────────────────────────────────────────────────────────────────
def save_html(
    tokens: List[str],
    H_vis: torch.Tensor,
    H_txt: torch.Tensor,
    output_path: str,
    question: str = "",
    cot_text: str = "",
    image=None,           # PIL Image，嵌入 HTML
    answer: str = "",
):
    VIG = (H_txt - H_vis).tolist()
    h_vis_list = H_vis.tolist()
    h_txt_list = H_txt.tolist()

    vig_max = max(max(VIG), 1e-6)
    vig_min = min(min(VIG), 0.0)

    # ── 图像 base64 ─────────────────────────────────────────────────────────────
    img_data_uri = _image_to_data_uri(image) if image is not None else ""
    img_block = (
        f'<img src="{img_data_uri}" alt="input image" '
        f'style="max-width:100%;max-height:340px;border-radius:8px;'
        f'box-shadow:0 2px 12px rgba(0,0,0,0.18);display:block;margin:0 auto;">'
        if img_data_uri else
        '<div style="width:100%;height:200px;background:#ddd;border-radius:8px;'
        'display:flex;align-items:center;justify-content:center;color:#888;font-size:14px;">'
        '（无图像）</div>'
    )

    def vig_to_color(vig_val: float) -> str:
        if vig_val < 0:
            t = min(abs(vig_val) / max(abs(vig_min), 1e-6), 1.0)
            r = int(150 - t * 50)
            g = int(150 - t * 50)
            b = int(220)
            a = 0.3 + t * 0.2
        else:
            t = min(vig_val / vig_max, 1.0)
            if t < 0.5:
                s = t * 2
                r = int(0   + s * 230)
                g = int(200)
                b = int(100 - s * 100)
            else:
                s = (t - 0.5) * 2
                r = int(230)
                g = int(200 - s * 150)
                b = int(0   + s * 50)
            a = 0.15 + t * 0.65
        return f"rgba({r},{g},{b},{a:.2f})"

    # ── token 着色 HTML（每个 token 带 data-idx 属性，供 JS 交互）─────────────
    token_html_parts = []
    for i, (tok, vig, hv, ht) in enumerate(zip(tokens, VIG, h_vis_list, h_txt_list)):
        color = vig_to_color(vig)
        tok_display = html_module.escape(tok).replace(" ", "&nbsp;").replace("\n", "↵<br>")
        token_html_parts.append(
            f'<span class="token" data-idx="{i}" data-vig="{vig:.4f}" '
            f'data-hv="{hv:.4f}" data-ht="{ht:.4f}" '
            f'style="background:{color}">{tok_display}</span>'
        )
    token_html = "".join(token_html_parts)

    # ── 统计 ────────────────────────────────────────────────────────────────────
    vig_arr = torch.tensor(VIG)
    total       = len(tokens)
    high_count  = (vig_arr > 0.5).sum().item()
    mid_count   = ((vig_arr > 0.2) & (vig_arr <= 0.5)).sum().item()
    low_count   = ((vig_arr >= 0) & (vig_arr <= 0.2)).sum().item()
    neg_count   = (vig_arr < 0).sum().item()
    mean_hvis   = H_vis.mean().item()
    mean_htxt   = H_txt.mean().item()
    mean_vig    = vig_arr.mean().item()
    pct_pos     = (vig_arr > 0).float().mean().item() * 100

    # ── top-10 高 VIG token 表格 ─────────────────────────────────────────────
    top_indices = vig_arr.argsort(descending=True)[:10].tolist()
    top_rows_html = ""
    for rank, idx in enumerate(top_indices):
        tok_repr = html_module.escape(repr(tokens[idx])[:22])
        hv  = h_vis_list[idx]
        ht  = h_txt_list[idx]
        vig = VIG[idx]
        bar_w = min(int(vig / vig_max * 80), 80)
        top_rows_html += (
            f'<tr><td>{rank+1}</td><td>{idx}</td>'
            f'<td class="tok-cell"><code>{tok_repr}</code></td>'
            f'<td>{hv:.3f}</td><td>{ht:.3f}</td>'
            f'<td><span style="color:#e74c3c;font-weight:bold">{vig:.3f}</span></td>'
            f'<td><div style="height:12px;width:{bar_w}px;background:linear-gradient(90deg,#f39c12,#e74c3c);'
            f'border-radius:3px;"></div></td></tr>\n'
        )

    # ── Chart.js 数据 ────────────────────────────────────────────────────────
    chart_step   = max(1, len(tokens) // 300)
    chart_labels = list(range(0, len(tokens), chart_step))
    chart_hvis   = [h_vis_list[i] for i in chart_labels]
    chart_htxt   = [h_txt_list[i] for i in chart_labels]
    chart_vig    = [VIG[i]        for i in chart_labels]

    # VIG 热力图 bar 数据（for histogram）
    hist_bins = 30
    hist_min, hist_max = float(vig_arr.min()), float(vig_arr.max())
    bin_w = (hist_max - hist_min) / hist_bins if hist_max > hist_min else 1.0
    hist_counts = [0] * hist_bins
    hist_bin_labels = []
    for b in range(hist_bins):
        lo = hist_min + b * bin_w
        hi = lo + bin_w
        hist_counts[b] = int(((vig_arr >= lo) & (vig_arr < hi)).sum().item())
        hist_bin_labels.append(f"{lo:.2f}")

    answer_line = f'<div class="answer-badge">✅ 正确答案：{html_module.escape(str(answer))}</div>' if answer else ""

    html_content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>VIG 可视化 — Token 级视觉信息增益</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* ── 全局 ── */
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: 'Segoe UI', 'PingFang SC', 'Helvetica Neue', sans-serif;
  margin: 0; padding: 0;
  background: #f0f2f5;
  color: #2c3e50;
}}
.page-wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px 20px; }}
/* ── 头部 ── */
.page-header {{
  background: linear-gradient(135deg, #1a237e 0%, #1565c0 60%, #0288d1 100%);
  color: white;
  border-radius: 16px;
  padding: 28px 32px 22px;
  margin-bottom: 24px;
  box-shadow: 0 4px 20px rgba(21,101,192,0.3);
}}
.page-header h1 {{ margin: 0 0 6px; font-size: 1.7em; letter-spacing: -0.3px; }}
.page-header p  {{ margin: 0; opacity: 0.85; font-size: 0.95em; }}
/* ── 卡片 ── */
.card {{
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.07);
  padding: 20px 24px;
  margin-bottom: 20px;
}}
.card-title {{
  font-size: 1.05em; font-weight: 600; color: #1565c0;
  margin: 0 0 14px;
  display: flex; align-items: center; gap: 8px;
}}
/* ── 顶部双栏布局 ── */
.top-grid {{
  display: grid;
  grid-template-columns: minmax(260px, 340px) 1fr;
  gap: 20px;
  margin-bottom: 20px;
  align-items: start;
}}
@media (max-width: 780px) {{ .top-grid {{ grid-template-columns: 1fr; }} }}
/* ── 图像面板 ── */
.image-panel {{
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.07);
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
.image-panel .panel-title {{
  font-size: 1em; font-weight: 600; color: #1565c0;
  display: flex; align-items: center; gap: 6px;
}}
.image-container {{
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  background: #f5f5f5;
  text-align: center;
}}
.answer-badge {{
  background: #e8f5e9;
  color: #2e7d32;
  border: 1px solid #a5d6a7;
  border-radius: 6px;
  padding: 6px 12px;
  font-size: 0.88em;
  font-weight: 500;
}}
/* ── 右侧信息面板 ── */
.info-panel {{
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.07);
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}}
.question-box {{
  background: #e3f2fd;
  border-left: 4px solid #1565c0;
  padding: 12px 16px;
  border-radius: 0 8px 8px 0;
  font-size: 0.95em;
  line-height: 1.6;
}}
/* ── 统计网格 ── */
.stats-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}}
@media (max-width: 600px) {{ .stats-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
.stat-card {{
  border-radius: 10px;
  padding: 14px 10px;
  text-align: center;
}}
.stat-card .val {{ font-size: 1.9em; font-weight: 700; line-height: 1.1; }}
.stat-card .lbl {{ font-size: 0.75em; color: #666; margin-top: 4px; line-height: 1.4; }}
.sc-red    {{ background: #fff5f5; }} .sc-red    .val {{ color: #e53935; }}
.sc-orange {{ background: #fff8f0; }} .sc-orange .val {{ color: #ef6c00; }}
.sc-green  {{ background: #f1f8e9; }} .sc-green  .val {{ color: #388e3c; }}
.sc-gray   {{ background: #f5f5f5; }} .sc-gray   .val {{ color: #757575; }}
/* ── 均值条 ── */
.mean-row {{
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  font-size: 0.88em;
}}
.mean-chip {{
  border-radius: 20px; padding: 4px 12px; font-weight: 600;
}}
.chip-blue   {{ background: #e3f2fd; color: #1565c0; }}
.chip-red    {{ background: #ffebee; color: #c62828; }}
.chip-green  {{ background: #e8f5e9; color: #2e7d32; }}
.chip-pct    {{ background: #f3e5f5; color: #6a1b9a; }}
/* ── 图例 ── */
.legend {{
  display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
  font-size: 0.85em;
}}
.legend-item {{ display: flex; align-items: center; gap: 5px; }}
.legend-box  {{ width: 16px; height: 16px; border-radius: 3px; flex-shrink: 0; }}
/* ── token 区 ── */
.token-area {{
  line-height: 2.0;
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
  font-size: 13.5px;
  max-height: 380px;
  overflow-y: auto;
  word-break: break-all;
  padding: 4px 2px;
  border-top: 1px solid #eee;
  margin-top: 10px;
}}
.token {{
  border-radius: 3px;
  padding: 1px 0;
  cursor: pointer;
  transition: outline 0.1s;
}}
.token:hover {{ outline: 2px solid #1565c0; outline-offset: 1px; }}
.token.active {{ outline: 2.5px solid #e53935; outline-offset: 1px; }}
/* ── 悬停详情框 ── */
#token-detail {{
  position: fixed;
  bottom: 20px; right: 20px;
  background: rgba(21,33,55,0.93);
  color: white;
  border-radius: 10px;
  padding: 12px 18px;
  font-size: 13px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.15s;
  z-index: 9999;
  min-width: 220px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  line-height: 1.8;
}}
#token-detail.show {{ opacity: 1; }}
/* ── Top-10 表格 ── */
.top-table {{ width: 100%; border-collapse: collapse; font-size: 0.87em; }}
.top-table th {{
  background: #1565c0; color: white;
  padding: 8px 10px; text-align: left; font-weight: 500;
}}
.top-table td {{ padding: 7px 10px; border-bottom: 1px solid #f0f0f0; }}
.top-table tr:hover td {{ background: #f5f8ff; }}
.tok-cell code {{ background: #f0f4ff; border-radius: 3px; padding: 1px 5px; }}
/* ── 折线图 ── */
.chart-wrap {{ padding: 20px 24px 10px; }}
/* ── 分布直方图 ── */
.hist-wrap {{ padding: 16px 24px 10px; }}
/* ── 原始文本 ── */
.raw-text {{
  font-family: monospace; font-size: 12.5px;
  background: #fafafa; border: 1px solid #e8e8e8;
  border-radius: 8px; padding: 14px 16px;
  white-space: pre-wrap; word-break: break-word;
  max-height: 180px; overflow-y: auto;
  color: #333;
  line-height: 1.7;
}}
/* ── 滚动条美化 ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: #f1f1f1; border-radius: 3px; }}
::-webkit-scrollbar-thumb {{ background: #bbb; border-radius: 3px; }}
</style>
</head>
<body>
<div class="page-wrap">

<!-- ── 头部 ── -->
<div class="page-header">
  <h1>🔍 VIG · Visual Information Gain — Token 级可视化</h1>
  <p>验证：去掉图像后生成 token 的熵是否升高 (H_txt &gt; H_vis)？哪些 token 最依赖视觉信息？</p>
</div>

<!-- ── 顶部双栏：图像 + 信息 ── -->
<div class="top-grid">

  <!-- 左：图像卡片 -->
  <div class="image-panel">
    <div class="panel-title">🖼️ 输入图像</div>
    <div class="image-container">
      {img_block}
    </div>
    {answer_line}
    <div style="font-size:0.8em;color:#888;text-align:center">
      总 token 数：<strong>{total}</strong> ｜
      VIG &gt; 0 比例：<strong style="color:#2e7d32">{pct_pos:.1f}%</strong>
    </div>
  </div>

  <!-- 右：问题 + 统计 -->
  <div class="info-panel">
    <div>
      <div style="font-weight:600;color:#1565c0;margin-bottom:6px">📋 问题</div>
      <div class="question-box">{html_module.escape(question[:400])}</div>
    </div>

    <div>
      <div style="font-weight:600;color:#1565c0;margin-bottom:10px">📊 Token 分布统计</div>
      <div class="stats-grid">
        <div class="stat-card sc-red">
          <div class="val">{high_count}</div>
          <div class="lbl">高 VIG<br>(VIG &gt; 0.5)<br>强视觉依赖</div>
        </div>
        <div class="stat-card sc-orange">
          <div class="val">{mid_count}</div>
          <div class="lbl">中 VIG<br>(0.2~0.5)<br>中等依赖</div>
        </div>
        <div class="stat-card sc-green">
          <div class="val">{low_count}</div>
          <div class="lbl">低 VIG<br>(0~0.2)<br>纯文本推理</div>
        </div>
        <div class="stat-card sc-gray">
          <div class="val">{neg_count}</div>
          <div class="lbl">负 VIG<br>(&lt; 0)<br>去图更确定</div>
        </div>
      </div>
    </div>

    <div>
      <div style="font-weight:600;color:#1565c0;margin-bottom:8px">📐 均值对比</div>
      <div class="mean-row">
        <span class="mean-chip chip-blue">H_vis = {mean_hvis:.4f} nats</span>
        <span class="mean-chip chip-red">H_txt = {mean_htxt:.4f} nats</span>
        <span class="mean-chip chip-green">ΔVIG = +{mean_vig:.4f} nats</span>
        <span class="mean-chip chip-pct">VIG&gt;0：{pct_pos:.1f}%</span>
      </div>
    </div>

    <div>
      <div style="font-weight:600;color:#1565c0;margin-bottom:6px">🎨 颜色图例</div>
      <div class="legend">
        <div class="legend-item"><div class="legend-box" style="background:rgba(230,50,50,0.75)"></div>高 VIG（红）</div>
        <div class="legend-item"><div class="legend-box" style="background:rgba(230,200,0,0.65)"></div>中 VIG（黄）</div>
        <div class="legend-item"><div class="legend-box" style="background:rgba(0,200,100,0.35)"></div>低 VIG（绿）</div>
        <div class="legend-item"><div class="legend-box" style="background:rgba(150,150,220,0.45)"></div>负 VIG（蓝灰）</div>
        <div style="color:#888;font-size:0.9em">← 点击 token 可高亮，悬停查看详细数值</div>
      </div>
    </div>
  </div>
</div><!-- /top-grid -->

<!-- ── Token 着色区 ── -->
<div class="card">
  <div class="card-title">🎨 Token 级 VIG 着色（生成文本全文）</div>
  <div class="token-area" id="tokenArea">{token_html}</div>
</div>

<!-- ── Top-10 高 VIG 表格 ── -->
<div class="card">
  <div class="card-title">🏆 Top-10 高 VIG Tokens（最依赖视觉信息）</div>
  <table class="top-table">
    <thead>
      <tr>
        <th>#</th><th>位置</th><th>Token 文本</th>
        <th>H_vis（带图）</th><th>H_txt（无图）</th>
        <th>VIG</th><th>强度</th>
      </tr>
    </thead>
    <tbody>
      {top_rows_html}
    </tbody>
  </table>
</div>

<!-- ── 熵折线图 ── -->
<div class="card">
  <div class="card-title">📈 逐 Token 熵对比曲线（带图 vs 无图）</div>
  <div class="chart-wrap" style="padding:0">
    <canvas id="entropyChart" height="75"></canvas>
  </div>
</div>

<!-- ── VIG 分布直方图 ── -->
<div class="card">
  <div class="card-title">📊 VIG 值分布直方图</div>
  <div class="hist-wrap" style="padding:0">
    <canvas id="histChart" height="55"></canvas>
  </div>
</div>

<!-- ── 原始生成文本 ── -->
<div class="card">
  <div class="card-title">📝 原始生成文本</div>
  <div class="raw-text">{html_module.escape(cot_text[:800])}</div>
</div>

</div><!-- /page-wrap -->

<!-- ── 悬停详情浮窗 ── -->
<div id="token-detail"></div>

<script>
// ── Chart.js：熵折线图 ──────────────────────────────────────────────────────
const entropyCtx = document.getElementById('entropyChart').getContext('2d');
new Chart(entropyCtx, {{
  type: 'line',
  data: {{
    labels: {chart_labels},
    datasets: [
      {{
        label: 'H_vis（带图熵）',
        data: {chart_hvis},
        borderColor: 'rgba(21,101,192,0.85)',
        backgroundColor: 'rgba(21,101,192,0.08)',
        borderWidth: 1.5, pointRadius: 0, fill: true,
      }},
      {{
        label: 'H_txt（无图熵）',
        data: {chart_htxt},
        borderColor: 'rgba(229,57,53,0.85)',
        backgroundColor: 'rgba(229,57,53,0.08)',
        borderWidth: 1.5, pointRadius: 0, fill: true,
      }},
      {{
        label: 'VIG = H_txt − H_vis',
        data: {chart_vig},
        borderColor: 'rgba(56,142,60,0.9)',
        backgroundColor: 'rgba(56,142,60,0.12)',
        borderWidth: 1, pointRadius: 0, fill: true,
        yAxisID: 'y2',
      }},
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      title: {{ display: true,
        text: 'H_txt（红）整体高于 H_vis（蓝）→ 去掉图像后模型更不确定，VIG 信号有效',
        font: {{ size: 13 }}, color: '#444'
      }},
      legend: {{ position: 'top', labels: {{ boxWidth: 14, font: {{ size: 12 }} }} }},
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Token 位置' }}, ticks: {{ maxTicksLimit: 15 }} }},
      y: {{ title: {{ display: true, text: '熵 (nats)' }}, position: 'left' }},
      y2: {{
        title: {{ display: true, text: 'VIG (nats)' }},
        position: 'right',
        grid: {{ drawOnChartArea: false }},
      }},
    }},
  }}
}});

// ── Chart.js：VIG 直方图 ────────────────────────────────────────────────────
const histCtx = document.getElementById('histChart').getContext('2d');
const histCounts = {hist_counts};
const histLabels = {hist_bin_labels};
const histColors = histLabels.map(v => {{
  const f = parseFloat(v);
  if (f < 0)   return 'rgba(150,150,220,0.65)';
  if (f < 0.2) return 'rgba(56,142,60,0.65)';
  if (f < 0.5) return 'rgba(239,108,0,0.65)';
  return 'rgba(229,57,53,0.75)';
}});
new Chart(histCtx, {{
  type: 'bar',
  data: {{
    labels: histLabels,
    datasets: [{{
      label: 'token 数量',
      data: histCounts,
      backgroundColor: histColors,
      borderRadius: 3,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true,
        text: 'VIG 分布：正态峰靠右 → 大多数 token 在无图时熵更高',
        font: {{ size: 13 }}, color: '#444'
      }},
      legend: {{ display: false }},
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'VIG 值 (nats)' }}, ticks: {{ maxTicksLimit: 12 }} }},
      y: {{ title: {{ display: true, text: 'token 数量' }} }},
    }},
  }}
}});

// ── Token 交互：点击高亮 + 悬停详情 ───────────────────────────────────────
const detailBox = document.getElementById('token-detail');
let activeToken = null;

document.querySelectorAll('.token').forEach(el => {{
  el.addEventListener('mouseenter', () => {{
    const idx = el.dataset.idx;
    const vig = parseFloat(el.dataset.vig);
    const hv  = parseFloat(el.dataset.hv);
    const ht  = parseFloat(el.dataset.ht);
    const vigColor = vig > 0.5 ? '#ff6b6b' : vig > 0.2 ? '#ffd93d' : vig < 0 ? '#aaa' : '#6bcb77';
    detailBox.innerHTML =
      `<div style="font-size:11px;opacity:.7;margin-bottom:4px">Token #${{idx}}</div>` +
      `<div style="font-family:monospace;font-size:15px;margin-bottom:8px;word-break:break-all">${{el.textContent}}</div>` +
      `<div>🔵 H<sub>vis</sub> <b>${{hv.toFixed(4)}}</b> nats &nbsp;（带图）</div>` +
      `<div>🔴 H<sub>txt</sub> <b>${{ht.toFixed(4)}}</b> nats &nbsp;（无图）</div>` +
      `<div style="margin-top:6px;font-size:14px">` +
      `VIG = <span style="color:${{vigColor}};font-weight:bold">${{vig >= 0 ? '+' : ''}}${{vig.toFixed(4)}}</span> nats` +
      (vig > 0.5 ? ' 🔴 强视觉依赖' : vig > 0.2 ? ' 🟡 中等依赖' : vig < 0 ? ' ⬜ 图像负影响' : ' 🟢 纯文本') +
      `</div>`;
    detailBox.classList.add('show');
  }});
  el.addEventListener('mouseleave', () => {{
    detailBox.classList.remove('show');
  }});
  el.addEventListener('click', () => {{
    if (activeToken) activeToken.classList.remove('active');
    if (activeToken === el) {{ activeToken = null; return; }}
    el.classList.add('active');
    activeToken = el;
  }});
}});
</script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n  ✅ HTML 可视化已保存: {output_path}")


# ── matplotlib 折线图 ──────────────────────────────────────────────────────────
def save_matplotlib_plot(
    tokens: List[str],
    H_vis: torch.Tensor,
    H_txt: torch.Tensor,
    output_path: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        VIG = (H_txt - H_vis).numpy()
        h_vis_np = H_vis.numpy()
        h_txt_np = H_txt.numpy()
        x = np.arange(len(tokens))

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        # 上图：H_vis vs H_txt
        axes[0].plot(x, h_vis_np, color="steelblue", linewidth=0.8, label="H_vis（带图）", alpha=0.9)
        axes[0].plot(x, h_txt_np, color="tomato",    linewidth=0.8, label="H_txt（无图）", alpha=0.9)
        axes[0].fill_between(x, h_vis_np, h_txt_np, where=(h_txt_np > h_vis_np),
                             alpha=0.2, color="green", label="VIG > 0 区域")
        axes[0].fill_between(x, h_vis_np, h_txt_np, where=(h_txt_np <= h_vis_np),
                             alpha=0.2, color="gray", label="VIG ≤ 0 区域")
        axes[0].set_ylabel("熵 (nats)", fontsize=11)
        axes[0].set_title("Token 级熵对比：H_vis vs H_txt（无图熵高于有图熵 → VIG > 0）", fontsize=12)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # 下图：VIG
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].bar(x, np.clip(VIG, 0, None), width=1, color="green",   alpha=0.5, label="正 VIG（视觉依赖）")
        axes[1].bar(x, np.clip(VIG, None, 0), width=1, color="gray",    alpha=0.5, label="负 VIG（忽略图像）")
        axes[1].set_xlabel("Token 位置", fontsize=11)
        axes[1].set_ylabel("VIG = H_txt - H_vis (nats)", fontsize=11)
        axes[1].set_title("VIG 分布（正值 → 模型需要图像信息）", fontsize=12)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Matplotlib 图已保存: {output_path}")
    except ImportError:
        print("  ⚠️  matplotlib 未安装，跳过 PNG 图")


# ── Demo 合成图像：带标注的三角形 ────────────────────────────────────────────
def _make_demo_image():
    """用 PIL 画一个带底边和高标注的三角形（模拟数学题图）。"""
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
        import io

        W, H = 400, 320
        img = PILImage.new("RGB", (W, H), color=(255, 255, 248))
        draw = ImageDraw.Draw(img)

        # 三角形顶点
        apex  = (200, 50)
        left  = (60, 250)
        right = (340, 250)

        # 填充（浅蓝色）
        draw.polygon([apex, left, right], fill=(220, 235, 255), outline=None)
        # 边框
        draw.line([apex, left, right, apex], fill=(40, 80, 160), width=3)

        # 高（虚线效果用短线段模拟）
        foot = (200, 250)
        for y in range(apex[1], foot[1], 10):
            draw.line([(200, y), (200, min(y + 6, foot[1]))], fill=(180, 60, 60), width=2)
        # 直角符号
        draw.rectangle([(196, 246), (204, 254)], outline=(180, 60, 60), width=1)

        # 文字标注
        try:
            font_big  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_mid  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
            font_sm   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            font_big = font_mid = font_sm = ImageFont.load_default()

        # 底边标注 "7 units"
        draw.line([(60, 270), (340, 270)], fill=(30, 30, 30), width=2)
        draw.line([(60, 263), (60, 277)], fill=(30, 30, 30), width=2)
        draw.line([(340, 263), (340, 277)], fill=(30, 30, 30), width=2)
        draw.text((175, 275), "7 units", font=font_big, fill=(20, 20, 20))

        # 高标注 "4 units"
        draw.text((210, 145), "4 units", font=font_mid, fill=(180, 60, 60))

        # 顶角标注
        draw.text((188, 28), "A", font=font_sm, fill=(60, 60, 200))
        draw.text((44, 252), "B", font=font_sm, fill=(60, 60, 200))
        draw.text((344, 252), "C", font=font_sm, fill=(60, 60, 200))

        # 底部问题提示
        draw.text((80, 295), "What is the length of the base BC?", font=font_sm, fill=(80, 80, 80))

        return img
    except Exception as e:
        print(f"  ⚠️  无法生成 demo 图像: {e}")
        return None


# ── Demo 模式（无需加载模型，使用合成数据验证可视化效果）─────────────────────
def run_demo(output_dir: str = "output"):
    """用合成数据演示可视化效果（无需 GPU / 模型）"""
    import numpy as np

    print(BOLD("\n=== DEMO 模式（合成数据）==="))
    print("生成合成 VIG 数据，模拟真实模型行为...\n")

    # 合成一段 CoT，包含视觉依赖 token 和纯文本 token
    question = "图中三角形的底边长为多少？(A) 5 (B) 7 (C) 9 (D) 11"
    answer   = "(B) 7"
    cot_text = (
        "The image shows a triangle with base and height marked. "
        "I can see the bottom edge labeled as 7 units. "
        "The height appears to be 4 units from the vertical line. "
        "Therefore the base length is 7. "
        "The answer is (B) 7."
    )

    # 生成合成图像
    demo_image = _make_demo_image()

    # 构造合成 token 序列
    words = cot_text.split()
    tokens = [" " + w for w in words]

    np.random.seed(42)

    # 模拟：视觉词汇（数字、空间词）→ 高 VIG；连接词、数学推理 → 低 VIG
    visual_keywords = {"7", "triangle", "base", "height", "bottom", "labeled", "4", "image", "see", "vertical", "marked"}
    text_keywords   = {"The", "I", "can", "from", "Therefore", "the", "is", "appears", "to", "be", "and", "of", "as"}

    H_vis, H_txt = [], []
    for tok in tokens:
        t = tok.strip().lower().rstrip(".,:")
        base_vis = np.random.uniform(1.5, 3.5)
        if t in visual_keywords:
            delta = np.random.uniform(0.8, 2.5)
        elif t in text_keywords:
            delta = np.random.uniform(-0.1, 0.15)
        else:
            delta = np.random.uniform(0.0, 0.5)
        H_vis.append(base_vis)
        H_txt.append(base_vis + delta)

    H_vis_t = torch.tensor(H_vis)
    H_txt_t = torch.tensor(H_txt)

    print_terminal_table(tokens, H_vis_t, H_txt_t, top_k=15)

    os.makedirs(output_dir, exist_ok=True)
    save_html(tokens, H_vis_t, H_txt_t,
              os.path.join(output_dir, "vig_demo.html"),
              question=question, cot_text=cot_text,
              image=demo_image, answer=answer)
    save_matplotlib_plot(tokens, H_vis_t, H_txt_t,
                         os.path.join(output_dir, "vig_demo.png"))


# ── 真实模型推理模式 ───────────────────────────────────────────────────────────
def run_real(args):
    """加载真实模型，对指定样本做 token 级 VIG 分析"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print(BOLD("\n=== 真实模型推理模式 ==="))
    print(f"  模型路径:  {args.model_path}")
    print(f"  数据集:    {args.dataset}")
    print(f"  样本索引:  {args.sample_idx}")
    print()

    # ── 加载模型 ────────────────────────────────────────────────────────────────
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration

    print("  加载 tokenizer & processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    print("  加载模型（float16，eval 模式）...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print("  模型加载完成。")

    # ── 加载数据集 ──────────────────────────────────────────────────────────────
    print(f"  加载数据集 {args.dataset}...")
    cache_dir = "./hf_cache/datasets"
    ds_map = {
        "mmstar":    ("Lin-Chen/MMStar", "val"),
        "mathvista": ("AI4Math/MathVista", "testmini"),
        "logicvista":("lscpku/LogicVista", "test"),
    }
    if args.dataset not in ds_map:
        print(f"  未知数据集: {args.dataset}，支持: {list(ds_map.keys())}")
        sys.exit(1)

    hf_id, split = ds_map[args.dataset]
    from datasets import load_dataset
    ds = load_dataset(hf_id, split=split, cache_dir=cache_dir)

    sample = ds[args.sample_idx]
    image    = sample.get("image", None)
    question = sample.get("question", "")
    options  = sample.get("options", None)
    answer   = sample.get("answer", "")

    if options:
        from data.dataset import _format_options
        question = f"{question}\nOptions: {_format_options(options)}"

    print(f"\n  问题: {question[:120]}...")
    print(f"  答案: {answer}")
    print()

    # ── 生成 CoT ────────────────────────────────────────────────────────────────
    if args.cot_text:
        cot_text = args.cot_text
        print(f"  使用指定 CoT（前 100 字符）: {cot_text[:100]}...")
    else:
        print("  生成 CoT 推理...")
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                top_p=0.95,
                do_sample=True,
            )
        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0, input_len:]
        cot_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"  生成完毕，长度: {len(gen_ids)} tokens")
        print(f"  前 200 字符: {cot_text[:200]}...")

    # ── 双向 forward 计算 token 级熵 ──────────────────────────────────────────
    print("\n  开始 token 级 VIG 计算...")
    tokens, H_vis, H_txt = compute_token_level_entropy(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        image=image,
        question=question,
        cot_text=cot_text,
        device="cuda",
    )

    # ── 输出 ────────────────────────────────────────────────────────────────────
    print_terminal_table(tokens, H_vis, H_txt, top_k=30)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, f"vig_{args.dataset}_{args.sample_idx}")

    save_html(tokens, H_vis, H_txt,
              prefix + ".html",
              question=question, cot_text=cot_text,
              image=image, answer=answer)

    save_matplotlib_plot(tokens, H_vis, H_txt, prefix + ".png")

    # 保存原始数值
    import json
    data_path = prefix + "_data.json"
    with open(data_path, "w") as f:
        json.dump({
            "question": question,
            "answer": answer,
            "cot_text": cot_text,
            "tokens": tokens,
            "H_vis": H_vis.tolist(),
            "H_txt": H_txt.tolist(),
            "VIG":   (H_txt - H_vis).tolist(),
        }, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 原始数据已保存: {data_path}")


# ── CLI 入口 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VIG 可视化工具")
    parser.add_argument("--demo", action="store_true",
                        help="快速演示模式（合成数据，无需 GPU）")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--dataset", type=str, default="mmstar",
                        choices=["mmstar", "mathvista", "logicvista"])
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="数据集中的样本索引")
    parser.add_argument("--cot_text", type=str, default="",
                        help="手动指定 CoT 文本（为空时模型自动生成）")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="生成 CoT 时的最大 token 数")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录")

    args = parser.parse_args()

    if args.demo:
        run_demo(output_dir=args.output_dir)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
