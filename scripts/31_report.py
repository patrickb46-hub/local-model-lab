#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31_report.py —— 把评测与训练记录渲染成一份自包含的可视化横评报告

设计取舍：
  * 图表全部用内联 SVG 现画，不引 CDN、不用 Chart.js —— 断网也能打开，
    双击即看，也方便直接打印/存档。
  * 浅色主题、克制的配色，四个模型四种色，一眼能分清谁是谁。
  * 先给结论，再给证据：顶部 KPI 卡 + 雷达图，往下才是逐项拆解。

用法：
  python 31_report.py
  python 31_report.py --open      # 生成后用默认浏览器打开
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, ensure_dir, lab_path, load_config, read_jsonl  # noqa: E402

PALETTE = {
    "base":    ("#6b7280", "base（Qwen3.5-2B 原版·HF）"),
    "A":       ("#2563eb", "A · 蒸馏 LoRA"),
    "A1":      ("#2563eb", "A1 · 蒸馏 60 步"),
    "A2":      ("#1d4ed8", "A2 · 蒸馏 120 步"),
    "B":       ("#ea580c", "B · 强化学习 LoRA"),
    "C":       ("#f59e0b", "C · SFT→RL（同预算对照）"),
    "dsv4pro": ("#0891b2", "DeepSeek V4 Pro（云 API）"),
    "qwen2b":  ("#0d9488", "官方 qwen3.5:2b（量化）"),
    "qwen4b":  ("#16a34a", "官方 qwen3.5:4b（量化）"),
    "teacher": ("#7c3aed", "teacher · 官方 qwen3.5:9b（量化）"),
}
# 分层定义（与 12_screen_layers.py 一致，判层只用 base 与 teacher 的测量，与训练方法无关）
LAYER_LABEL = {
    "L1": "L1 · teacher 会做",
    "L2": "L2 · teacher 系统性失败（主战场）",
    "L3": "L3 · 谁都做不出",
}
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"


# ---------------------------------------------------------------------------
# SVG 组件
# ---------------------------------------------------------------------------

def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def grouped_bars(metrics, metric_keys, labels, title, width=760, height=300):
    """分组柱状图：每个指标一组，组内每个模型一根柱子。"""
    roles = [r for r in PALETTE if r in metrics["models"]]
    if not roles:
        return "<p>无数据</p>"
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 60
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    n_grp = len(metric_keys)
    gw = iw / n_grp
    bw = min(34, gw / (len(roles) + 1))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px" role="img">']
    parts.append(f'<text x="{width/2}" y="18" text-anchor="middle" '
                 f'font-size="13" font-weight="600" fill="{INK}">{esc(title)}</text>')
    # 网格与刻度
    for i in range(6):
        y = pad_t + ih * i / 5
        v = 100 - i * 20
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+iw}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" text-anchor="end" '
                     f'font-size="10" fill="{MUTED}">{v}%</text>')
    # 柱子
    for gi, (mk, lab) in enumerate(zip(metric_keys, labels)):
        gx = pad_l + gi * gw
        base_y = pad_t + ih
        for ri, role in enumerate(roles):
            v = metrics["models"][role].get(mk)
            if v is None:
                continue
            v = max(0.0, min(1.0, float(v)))
            x = gx + gw / 2 - (len(roles) * bw) / 2 + ri * bw
            h = ih * v
            color = PALETTE[role][0]
            parts.append(f'<rect x="{x:.1f}" y="{base_y-h:.1f}" width="{bw-4:.1f}" '
                         f'height="{h:.1f}" fill="{color}" rx="2"/>')
            if h > 16:
                parts.append(f'<text x="{x+(bw-4)/2:.1f}" y="{base_y-h+13:.1f}" '
                             f'text-anchor="middle" font-size="9.5" fill="#ffffff" '
                             f'font-weight="600">{v*100:.0f}</text>')
            else:
                parts.append(f'<text x="{x+(bw-4)/2:.1f}" y="{base_y-h-4:.1f}" '
                             f'text-anchor="middle" font-size="9.5" fill="{color}" '
                             f'font-weight="600">{v*100:.0f}</text>')
        parts.append(f'<text x="{gx+gw/2:.1f}" y="{base_y+16:.1f}" text-anchor="middle" '
                     f'font-size="11" fill="{INK}">{esc(lab)}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+ih}" x2="{pad_l+iw}" y2="{pad_t+ih}" '
                 f'stroke="#9ca3af" stroke-width="1.2"/>')
    # 图例
    lx = pad_l
    for role in roles:
        parts.append(f'<rect x="{lx}" y="{height-22}" width="10" height="10" '
                     f'fill="{PALETTE[role][0]}" rx="2"/>')
        parts.append(f'<text x="{lx+14}" y="{height-13}" font-size="10.5" '
                     f'fill="{MUTED}">{esc(PALETTE[role][1])}</text>')
        lx += 16 + len(PALETTE[role][1]) * 6.6
    parts.append("</svg>")
    return "".join(parts)


def radar(metrics, axes, size=380):
    """雷达图：五个维度看四模型的能力形状。"""
    roles = [r for r in PALETTE if r in metrics["models"]]
    if not roles:
        return "<p>无数据</p>"
    import math
    cx = cy = size / 2
    R = size / 2 - 62
    n = len(axes)
    parts = [f'<svg viewBox="0 0 {size} {size}" width="100%" style="max-width:{size}px">']
    parts.append(f'<text x="{cx}" y="16" text-anchor="middle" font-size="13" '
                 f'font-weight="600" fill="{INK}">能力形状对比</text>')
    # 环
    for k in range(1, 5):
        r = R * k / 4
        pts = " ".join(
            f"{cx + r*math.cos(-math.pi/2 + 2*math.pi*i/n):.1f},"
            f"{cy + r*math.sin(-math.pi/2 + 2*math.pi*i/n):.1f}" for i in range(n))
        parts.append(f'<polygon points="{pts}" fill="none" stroke="{GRID}"/>')
    # 轴与标签
    for i, ax in enumerate(axes):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        x, y = cx + R * math.cos(ang), cy + R * math.sin(ang)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
                     f'stroke="{GRID}"/>')
        lx, ly = cx + (R + 22) * math.cos(ang), cy + (R + 22) * math.sin(ang)
        anchor = "middle"
        if math.cos(ang) > 0.3:
            anchor = "start"
        elif math.cos(ang) < -0.3:
            anchor = "end"
        parts.append(f'<text x="{lx:.1f}" y="{ly+4:.1f}" text-anchor="{anchor}" '
                     f'font-size="10.5" fill="{MUTED}">{esc(ax[1])}</text>')
    # 数据多边形
    for role in roles:
        m = metrics["models"][role]
        pts = []
        for i, (key, _) in enumerate(axes):
            v = m.get(key)
            v = 0.0 if v is None else max(0.0, min(1.0, float(v)))
            ang = -math.pi / 2 + 2 * math.pi * i / n
            pts.append(f"{cx + R*v*math.cos(ang):.1f},{cy + R*v*math.sin(ang):.1f}")
        color = PALETTE[role][0]
        parts.append(f'<polygon points="{" ".join(pts)}" fill="{color}" '
                     f'fill-opacity="0.13" stroke="{color}" stroke-width="1.8"/>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(series, title, width=760, height=250, y_label="", y_min=None, y_max=None):
    """折线图。series = [(label, color, [values]), ...]"""
    series = [s for s in series if s[2]]
    if not series:
        return "<p>暂无训练记录</p>"
    pad_l, pad_r, pad_t, pad_b = 62, 24, 30, 44
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    allv = [v for s in series for v in s[2]]
    lo = y_min if y_min is not None else min(allv)
    hi = y_max if y_max is not None else max(allv)
    if hi - lo < 1e-9:
        hi, lo = lo + 1, lo
    n = max(len(s[2]) for s in series)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px">']
    parts.append(f'<text x="{width/2}" y="18" text-anchor="middle" font-size="13" '
                 f'font-weight="600" fill="{INK}">{esc(title)}</text>')
    for i in range(5):
        y = pad_t + ih * i / 4
        v = hi - (hi - lo) * i / 4
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+iw}" y2="{y:.1f}" '
                     f'stroke="{GRID}"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                     f'fill="{MUTED}">{v:.2f}</text>')
    for label, color, vals in series:
        step = iw / max(n - 1, 1)
        pts = [(pad_l + i * step, pad_t + ih * (1 - (v - lo) / (hi - lo)))
               for i, v in enumerate(vals)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}"/>')
    lx = pad_l
    for label, color, _ in series:
        parts.append(f'<rect x="{lx}" y="{height-20}" width="10" height="3" fill="{color}"/>')
        parts.append(f'<text x="{lx+14}" y="{height-14}" font-size="10.5" '
                     f'fill="{MUTED}">{esc(label)}</text>')
        lx += 16 + len(label) * 7.0
    parts.append(f'<text x="{pad_l}" y="{height-4}" font-size="10" fill="{MUTED}">'
                 f'{esc(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def hbars(data, title, width=760, row_h=26, label_w=96):
    """横向条形图：data = [(label, value, color, note), ...]"""
    if not data:
        return "<p>无数据</p>"
    height = 34 + row_h * len(data) + 16
    iw = width - label_w - 90
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px">']
    parts.append(f'<text x="{width/2}" y="18" text-anchor="middle" font-size="13" '
                 f'font-weight="600" fill="{INK}">{esc(title)}</text>')
    for i, (label, v, color, note) in enumerate(data):
        y = 30 + i * row_h
        w = max(1.0, iw * max(0.0, min(1.0, v)))
        parts.append(f'<text x="{label_w-8}" y="{y+13}" text-anchor="end" '
                     f'font-size="11" fill="{INK}">{esc(label)}</text>')
        parts.append(f'<rect x="{label_w}" y="{y+3}" width="{iw}" height="14" '
                     f'fill="#f3f4f6" rx="3"/>')
        parts.append(f'<rect x="{label_w}" y="{y+3}" width="{w:.1f}" height="14" '
                     f'fill="{color}" rx="3"/>')
        parts.append(f'<text x="{label_w+iw+8}" y="{y+14}" font-size="10.5" '
                     f'fill="{MUTED}">{v*100:.0f}% {esc(note)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 报告组装
# ---------------------------------------------------------------------------

def kpi_card(role, m, baseline):
    color, label = PALETTE[role]
    def delta(key):
        if baseline is None or role == "base" or key not in baseline:
            return ""
        d = (m.get(key) or 0) - (baseline.get(key) or 0)
        if abs(d) < 0.005:
            return '<span class="flat">±0</span>'
        arrow = "▲" if d > 0 else "▼"
        cls = "up" if d > 0 else "down"
        return f'<span class="{cls}">{arrow}{abs(d)*100:.0f}pt</span>'
    return f"""
    <div class="kpi" style="--c:{color}">
      <div class="kpi-title" style="color:{color}">{esc(label)}</div>
      <div class="kpi-row"><span class="k">格式遵从率</span>
        <span class="v">{(m.get('format_rate') or 0)*100:.0f}%</span>{delta('format_rate')}</div>
      <div class="kpi-row"><span class="k">任务正确率</span>
        <span class="v">{(m.get('accuracy') or 0)*100:.0f}%</span>{delta('accuracy')}</div>
      <div class="kpi-row"><span class="k">与 teacher 一致性</span>
        <span class="v">{(m.get('consistency') or 0)*100:.0f}%</span>{delta('consistency')}</div>
      <div class="kpi-row"><span class="k">输出多样性</span>
        <span class="v">{(m.get('diversity') or 0)*100:.0f}%</span>{delta('diversity')}</div>
      <div class="kpi-row"><span class="k">通用能力保持</span>
        <span class="v">{('—' if m.get('general_probe') is None else f"{m['general_probe']*100:.0f}%")}</span></div>
      <div class="kpi-foot">均步数 {m.get('avg_steps','—')} · 均耗时 {m.get('avg_latency_s','—')}s · {m.get('n_main',0)} 题</div>
    </div>"""


def layer_bars(metrics, width=760, height=310):
    """分层正确率分组柱状图（L1/L2/L3 各一组）。

    为什么单列一张图：总分会被题目难度构成掩盖。本实验的核心命题是
    「蒸馏被 teacher 封顶、RL 能突破」——只有拆到层上才看得见。
    """
    flat = {"models": {}}
    for r, m in metrics["models"].items():
        d = dict(m)
        for lay in ("L1", "L2", "L3"):
            v = (m.get("by_layer") or {}).get(lay)
            d[f"acc_{lay}"] = (v["accuracy"] if v else None)
        flat["models"][r] = d
    return grouped_bars(flat, ["acc_L1", "acc_L2", "acc_L3"],
                        [LAYER_LABEL[k] for k in ("L1", "L2", "L3")],
                        "分层正确率（%）— 按 teacher 会不会做切分", width, height)


def layer_section(metrics, roles) -> str:
    """分层对比表：本实验的主指标。"""
    ms = metrics["models"]
    present = [l for l in ("L1", "L2", "L3")
               if any((ms[r].get("by_layer") or {}).get(l) for r in roles)]
    if not present:
        return ("<p>本轮未使用分层基准（<code>shared/bench_stratified.jsonl</code>），"
                "因此没有分层数据。</p>")
    reps = max(1, ms[roles[0]].get("repeats", 1) or 1)
    nq = {l: (ms[roles[0]]["by_layer"].get(l, {}).get("n", 0) // reps)
          for l in present}
    head = "".join(
        f"<th>{esc(LAYER_LABEL[l])}<br><span class='flat'>n={nq[l]} 题</span></th>"
        for l in present) + "<th>总体（95% CI）</th>"
    body = ""
    for r in roles:
        m = ms[r]
        tds = ""
        for l in present:
            v = (m.get("by_layer") or {}).get(l)
            tds += f"<td>{v['accuracy']*100:.1f}%</td>" if v else "<td>—</td>"
        ci = m.get("accuracy_ci") or []
        ci_s = (f"<br><span class='flat'>[{ci[0]*100:.1f}, {ci[1]*100:.1f}]</span>"
                if len(ci) == 2 else "")
        body += (f'<tr><td class="role" style="color:{PALETTE[r][0]}">'
                 f'{esc(PALETTE[r][1])}</td>{tds}'
                 f"<td><b>{m['accuracy']*100:.1f}%</b>{ci_s}</td></tr>")
    return (f"<table><thead><tr><th>模型</th>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>")


def significance_section(metrics) -> str:
    """配对显著性检验表（McNemar 精确检验）。"""
    pw = metrics.get("pairwise") or {}
    if not pw:
        return "<p>无配对检验数据。</p>"
    prereg = [
        ("A1_vs_B", "A1 vs B —— <b>主对比</b>：60:60 严格等步数，唯一变量是「模仿 vs 验证器」"),
        ("C_vs_A2", "C vs A2 —— <b>等总预算对照</b>：同为 120 步，纯 SFT 打得过「SFT+RL」吗"),
        ("B_vs_teacher", "B vs teacher —— RL 能不能真正越过教师天花板"),
        ("A1_vs_teacher", "A1 vs teacher —— 蒸馏是否被教师结构性封顶"),
        ("C_vs_B", "C vs B —— 蒸馏预热对 RL 有没有帮助"),
    ]
    rows = ""

    def _row(k, desc):
        v = pw.get(k)
        if not v:
            return ""
        sig = "★ 显著" if v["significant"] else "不显著"
        style = ' style="background:#f0fdf4"' if v["significant"] else ""
        return (f"<tr{style}><td>{desc}</td><td>{v['only_a']}</td>"
                f"<td>{v['only_b']}</td><td>{v['n_common']}</td>"
                f"<td>{v['p_value']}</td><td>{sig}</td></tr>")

    for k, desc in prereg:
        rows += _row(k, desc)
    for k in sorted(pw):
        if k in dict(prereg):
            continue
        a, b = k.split("_vs_")
        rows += _row(k, f"{esc(a)} vs {esc(b)}")
    return ("<div class='note'>逐题多数投票二值化（该题多次采样中过半答对 → 记为「会做」），"
            "再做 McNemar 精确检验。<b>配对</b>的意义：两个模型做的是同一批题，"
            "题目难度是共同变量；不配对会把「题目难」误算成「模型差」。</div>"
            "<table><thead><tr><th>对比</th><th>仅前者会</th><th>仅后者会</th>"
            "<th>共同题</th><th>p 值</th><th>结论</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def conclusion(metrics, roles) -> str:
    ms = metrics["models"]
    pw = metrics.get("pairwise") or {}
    if not ms:
        return "<p>还没有评测数据。</p>"

    # 动态识别各臂：兼容新命名（A1/A2/B/C）与旧命名（A/B）
    A_MAIN = next((r for r in ("A1", "A") if r in ms), None)
    A_LONG = "A2" if "A2" in ms else None
    B_ARM = "B" if "B" in ms else None
    C_ARM = "C" if "C" in ms else None
    NAME = {"A1": "A1 · 蒸馏 60 步", "A": "A · 蒸馏",
            "A2": "A2 · 蒸馏 120 步", "B": "B · 强化学习",
            "C": "C · SFT→RL（同预算）"}
    arms = [r for r in (A_MAIN, A_LONG, B_ARM, C_ARM) if r]
    base = ms.get("base")
    tea = ms.get("teacher")
    lines = []

    def _lay(r, lay):
        v = (ms.get(r, {}).get("by_layer") or {}).get(lay)
        return v["accuracy"] if v else None

    has_layer = any((ms[r].get("by_layer") or {}) for r in ms)

    # ---- ① 核心命题：在 teacher 系统性失败的题上，谁突破了天花板 ----
    if has_layer and tea and _lay("teacher", "L2") is not None:
        lt = _lay("teacher", "L2")
        cells = []
        for r in arms:
            v = _lay(r, "L2")
            if v is not None:
                mark = " ↑" if v > lt else ""
                cells.append(f"{NAME.get(r, r)} <b>{v*100:.1f}%</b>{mark}")
        beat_rl = [r for r in (B_ARM, C_ARM) if r and (_lay(r, "L2") or 0) > lt]
        beat_dist = [r for r in (A_MAIN, A_LONG) if r and (_lay(r, "L2") or 0) > lt]
        if beat_rl and not beat_dist:
            verdict = ("<b>命题成立</b>：RL 臂越过了 teacher，而蒸馏臂没有。"
                       "原因在数据管线 —— 蒸馏剔除了 teacher 答错的样本，"
                       "它<b>从未见过这类题的正确解法</b>；RL 只需要一个可验证的判分器")
        elif beat_rl and beat_dist:
            verdict = "<b>两类方法都越过了 teacher</b>（这一层的区分度不足，需结合差值与显著性判读）"
        elif beat_dist and not beat_rl:
            verdict = ("<b>与命题相反</b>：反而是蒸馏越过了 teacher —— "
                       "需要复查 L2 的定义是否真的满足「teacher 系统性失败」")
        else:
            verdict = "<b>都没有越过 teacher</b> —— 这一层没有区分出方法差异"
        lines.append(
            f"<li><b>核心命题 · L2 层（teacher 系统性失败、base 尚有机会的题）</b>："
            f"teacher {lt*100:.1f}%；" + "；".join(cells) + "。" + verdict + "。</li>")

        # L1 层：蒸馏的主场
        lt1 = _lay("teacher", "L1")
        if lt1 is not None:
            d_cells = []
            for r in (A_MAIN, A_LONG, B_ARM, C_ARM):
                v = _lay(r, "L1") if r else None
                if v is not None:
                    d_cells.append(f"{NAME.get(r, r)} {v*100:.1f}%")
            if d_cells:
                lines.append(
                    f"<li><b>对照 · L1 层（teacher 会做的题）</b>：teacher {lt1*100:.1f}%；"
                    + "；".join(d_cells) +
                    "。这是蒸馏的主场（它学的就是这些），若此处蒸馏并未领先，"
                    "说明它的模仿没有真正生效。</li>")

    # ---- ② 各臂相对 base 的总增益 ----
    if base:
        for r in arms:
            m = ms[r]
            d_acc = (m["accuracy"] - base["accuracy"]) * 100
            d_fmt = (m["format_rate"] - base["format_rate"]) * 100
            verdict = "有增益" if d_acc > 1 else ("基本持平" if abs(d_acc) <= 1 else "反而更差")
            lines.append(
                f"<li><b>{NAME.get(r, r)}</b>：总体正确率相对 base <b>{d_acc:+.1f}pt</b>，"
                f"格式遵从率 <b>{d_fmt:+.1f}pt</b> —— {verdict}。"
                f"多样性 {m['diversity']*100:.0f}%，与 teacher 一致性 {m['consistency']*100:.0f}%。</li>")

    # ---- ③ 预注册对比的显著性 ----
    for k, desc in [("A1_vs_B", "A1 vs B（严格等步数，唯一变量是「模仿 vs 验证器」）"),
                    ("C_vs_A2", "C vs A2（同为 120 步，等总预算）"),
                    ("B_vs_teacher", "B vs teacher")]:
        v = pw.get(k)
        if not v:
            continue
        a, b = k.split("_vs_")
        sig = "显著" if v["significant"] else "不显著"
        lines.append(
            f"<li><b>{desc}</b>：只有 {a} 会的 {v['only_a']} 题、只有 {b} 会的 "
            f"{v['only_b']} 题（共 {v['n_common']} 题）→ McNemar p={_fmt_p(v['p_value'])}，"
            f"<b>{sig}</b>。</li>")

    # ---- ④ 微调的 2B 能顶上原版 4B 吗 ----
    q4 = ms.get("qwen4b")
    if q4 and arms:
        best = max(arms, key=lambda r: ms[r]["accuracy"])
        gap = (q4["accuracy"] - ms[best]["accuracy"]) * 100
        if gap <= 2:
            v = ("<b>追上了</b> —— 微调后的小模型已可替代原版 4B 干这类活，"
                 "省下的是内存与推理时间")
        elif gap <= 10:
            v = f"<b>接近但还差一口气</b>（差 {gap:.1f}pt）"
        else:
            v = f"<b>差距明显</b>（差 {gap:.1f}pt）"
        lines.append(
            f"<li><b>微调的 2B 能顶上原版 4B 吗？</b>最好的臂是 <b>{NAME.get(best, best)}</b>"
            f"（{ms[best]['accuracy']*100:.1f}%），官方 qwen3.5:4b 是 "
            f"{q4['accuracy']*100:.1f}% —— {v}。</li>")

    # ---- ⑤ 外部强模型参照 ----
    cl = ms.get("dsv4pro")
    if cl and arms:
        best = max(arms, key=lambda r: ms[r]["accuracy"])
        gap = (cl["accuracy"] - ms[best]["accuracy"]) * 100
        if gap <= 0:
            v = "<b>已经追平甚至反超</b>"
        elif gap <= 10:
            v = f"<b>已经很接近</b>（差 {gap:.1f}pt）"
        elif gap <= 25:
            v = f"<b>还有一段路</b>（差 {gap:.1f}pt）"
        else:
            v = f"<b>差距明显</b>（差 {gap:.1f}pt）"
        lines.append(
            f"<li><b>外部强模型参照（DeepSeek V4 Pro · 云 API）</b>："
            f"{cl['accuracy']*100:.1f}%，最好的臂 <b>{NAME.get(best, best)}</b> "
            f"{ms[best]['accuracy']*100:.1f}% —— {v}。"
            f"顺带一个诚实的观察：云 API 单条 {cl['avg_latency_s']:.1f}s，"
            f"本机单条 {ms[best]['avg_latency_s']:.1f}s —— 在这台没有 GPU 的机器上，"
            f"本地推理慢一个数量级。</li>")

    # ---- ⑥ 遗忘 / 坍缩警告 ----
    for r in arms:
        m = ms[r]
        if m.get("general_probe") is not None and m["general_probe"] < 0.5:
            lines.append(f"<li class='warn'><b>注意</b>：{NAME.get(r, r)} 的通用能力探针命中率仅 "
                         f"{m['general_probe']*100:.0f}%，存在灾难性遗忘迹象。</li>")
        if m["diversity"] < 0.05 and m["n_questions"] >= 8:
            lines.append(f"<li class='warn'><b>注意</b>：{NAME.get(r, r)} 的输出多样性仅 "
                         f"{m['diversity']*100:.0f}%，输出可能已趋同（训崩的前兆）。</li>")
    return "<ul>" + "".join(lines) + "</ul>"


def think_section(metrics) -> str:
    """thinking 开 / 关 的对照表。"""
    tm = metrics.get("think") or {}
    ms = metrics["models"]
    if not tm:
        return "<p>未采集 thinking 对照数据。</p>"
    roles = [r for r in PALETTE if r in tm or r in ms]
    rows = []
    for r in roles:
        off = ms.get(r)
        on = tm.get(r)
        if not on and not off:
            continue
        def cell(m, k, pct=True):
            if not m or m.get(k) is None:
                return "—"
            v = m[k]
            return f"{v*100:.1f}%" if pct else f"{v:.0f}"
        rows.append(
            f'<tr><td class="role" style="color:{PALETTE[r][0]}">{esc(PALETTE[r][1])}</td>'
            f'<td>{cell(off,"format_rate")}</td><td>{cell(on,"format_rate")}</td>'
            f'<td>{cell(off,"accuracy")}</td><td>{cell(on,"accuracy")}</td>'
            f'<td>{cell(on,"avg_thinking_chars",False)}</td>'
            f'<td>{cell(on,"avg_latency_s",False)}</td></tr>')
    return f"""<table><thead><tr>
      <th>模型</th><th>格式·关</th><th>格式·开</th>
      <th>正确·关</th><th>正确·开</th><th>开时推理字数</th><th>开时耗时(s)</th>
    </tr></thead><tbody>{''.join(rows)}</tbody></table>"""


def build_html(cfg, metrics, dist_log, rl_log) -> str:
    roles = [r for r in PALETTE if r in metrics["models"]]
    ms = metrics["models"]
    baseline = ms.get("base")

    axes = [("format_rate", "格式遵从"), ("accuracy", "正确率"),
            ("consistency", "学得像 teacher"), ("diversity", "多样性"),
            ("general_probe", "通用能力")]

    kpis = "".join(kpi_card(r, ms[r], baseline) for r in roles)

    bars = grouped_bars(
        metrics,
        ["format_rate", "accuracy", "consistency", "diversity"],
        ["格式遵从率", "任务正确率", "与 teacher 一致性", "输出多样性"],
        "四项核心指标（%）")
    rd = radar(metrics, axes)

    # 分层视图 —— 本实验的主指标（总分会被题目难度构成掩盖）
    layer_svg = layer_bars(metrics)
    lyr = layer_section(metrics, roles)
    sig = significance_section(metrics)

    # 分类别正确率：以 base 为基准排
    cats = {}
    for r in roles:
        for c, v in ms[r].get("by_category", {}).items():
            cats.setdefault(c, {})[r] = v["accuracy"]
    cat_rows = []
    for c in sorted(cats, key=lambda k: -(cats[k].get("A", 0) + cats[k].get("B", 0))):
        for r in roles:
            if r in cats[c]:
                cat_rows.append((f"{c} · {r}", cats[c][r], PALETTE[r][0], ""))
    cat_svg = hbars(cat_rows, "分类别正确率", row_h=22)

    # 训练曲线
    dist_loss = [r["loss"] for r in dist_log]
    dist_svg = line_chart([("SFT loss", "#2563eb", dist_loss)], "支线 A · 蒸馏训练 loss",
                          y_label="loss")
    rl_reward = [r["mean_reward"] for r in rl_log]
    rl_acc = [r["accuracy"] for r in rl_log]
    rl_fmt = [r["format_rate"] for r in rl_log]
    rl_svg = line_chart([
        ("平均奖励", "#ea580c", rl_reward),
        ("正确率", "#2563eb", rl_acc),
        ("格式遵从率", "#7c3aed", rl_fmt),
    ], "支线 B · GRPO 奖励曲线", y_label="step")

    def table_rows():
        cols = ["格式遵从率", "任务正确率", "与 teacher 一致性", "输出多样性",
                "通用能力保持", "均步数", "均耗时(s)", "题数"]
        keys = ["format_rate", "accuracy", "consistency", "diversity",
                "general_probe", "avg_steps", "avg_latency_s", "n_main"]
        rate_keys = ("format_rate", "accuracy", "consistency", "diversity",
                     "general_probe")
        head = "".join(f"<th>{c}</th>" for c in cols)
        body = ""
        for r in roles:
            m = ms[r]
            tds = ""
            for k in keys:
                v = m.get(k)
                if v is None:
                    tds += "<td>—</td>"
                elif isinstance(v, float) and k in rate_keys:
                    sd = m.get(k + "_std")
                    if sd:
                        tds += f"<td>{v*100:.1f}<span class='pm'>±{sd*100:.1f}</span>%</td>"
                    else:
                        tds += f"<td>{v*100:.1f}%</td>"
                else:
                    tds += f"<td>{v}</td>"
            body += (f'<tr><td class="role" style="color:{PALETTE[r][0]}">'
                     f'{esc(PALETTE[r][1])}</td>{tds}</tr>')
        return f'<table><thead><tr><th>模型</th>{head}</tr></thead><tbody>{body}</tbody></table>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>本地模型实验室 · 蒸馏 vs 强化学习 横评</title>
<style>
  :root {{ --ink:{INK}; --muted:{MUTED}; --grid:{GRID}; --card:#ffffff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px 20px 60px; background:#f6f7f9; color:var(--ink);
         font:15px/1.65 -apple-system,"PingFang SC","Helvetica Neue",Arial,sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:.3px; }}
  .sub {{ color:var(--muted); font-size:13.5px; margin-bottom:26px; }}
  .card {{ background:var(--card); border:1px solid var(--grid); border-radius:12px;
           padding:22px 24px; margin-bottom:20px; }}
  h2 {{ font-size:17px; margin:0 0 14px; padding-bottom:9px;
        border-bottom:1px solid var(--grid); }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(228px,1fr)); gap:14px; }}
  .kpi {{ background:#fff; border:1px solid var(--grid); border-left:4px solid var(--c);
          border-radius:10px; padding:14px 16px; }}
  .kpi-title {{ font-weight:700; font-size:13.5px; margin-bottom:9px; }}
  .kpi-row {{ display:flex; justify-content:space-between; font-size:12.5px;
              padding:3px 0; border-bottom:1px dashed #f1f2f4; }}
  .kpi-row:last-of-type {{ border-bottom:none; }}
  .kpi-row .k {{ color:var(--muted); }}
  .kpi-row .v {{ font-weight:700; font-variant-numeric:tabular-nums; }}
  .kpi-foot {{ margin-top:9px; font-size:11.5px; color:var(--muted); }}
  .up {{ color:#b91c1c; font-size:11.5px; margin-left:7px; font-weight:600; }}
  .down {{ color:#15803d; font-size:11.5px; margin-left:7px; font-weight:600; }}
  .flat {{ color:var(--muted); font-size:11.5px; margin-left:7px; }}
  .pm {{ color:var(--muted); font-size:10.5px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:9px 10px; text-align:right; border-bottom:1px solid var(--grid);
           font-variant-numeric:tabular-nums; }}
  th {{ background:#fafbfc; color:var(--muted); font-weight:600; font-size:12px; }}
  th:first-child, td:first-child {{ text-align:left; }}
  td.role {{ font-weight:700; }}
  ul {{ margin:6px 0 0; padding-left:20px; }}
  li {{ margin-bottom:7px; }}
  li.warn {{ color:#b45309; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  @media (max-width:820px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  .note {{ font-size:12.5px; color:var(--muted); margin-top:12px; line-height:1.6; }}
  code {{ background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12px; }}
</style></head><body><div class="wrap">

<h1>本地模型实验室 · 蒸馏 vs 强化学习</h1>
<div class="sub">
  同一份 Qwen3.5-2B 复制成两条互不干扰的支线 ·
  统一任务轴：初中数学小题 + 固定结构化输出格式 ·
  评测集 {metrics.get('n_eval','—')} 题（训练前锁定，两条支线均未见过）
</div>

<div class="card">
  <h2>一、结论先行</h2>
  {conclusion(metrics, roles)}
  <div class="note">
    读法：<b>蒸馏支线 A</b> 看"学得像不像 teacher"，<b>强化学习支线 B</b> 看"奖励曲线是否上行"。
    两者过程指标本就不同源，只有在下游同一套题上的数字才可直接同框比较。
  </div>
</div>

<div class="card">
  <h2>二、各模型的关键指标</h2>
  <div class="kpis">{kpis}</div>
  <div class="note">▲/▼ 表示相对 base 的百分点变化。</div>
</div>

<div class="card">
  <h2>三、横向对比（总体）</h2>
  {bars}
</div>

<div class="card">
  <h2>四、分层对比 —— 本实验的主指标</h2>
  {layer_svg}
  {lyr}
  <div class="note">
    分层<b>只用 base 与 teacher 的测量判定，与训练方法完全无关</b> ——
    所以它不构成「挑对自己有利的题」。用的题目在训练前就已冻结。<br>
    <b>L1</b>：teacher 会做 → 蒸馏的主场（它学的就是这些）。<br>
    <b>L2</b>：teacher 系统性失败、而 base 尚有可能做对 → <b>RL 的战场</b>。
    蒸馏在数据管线上有 <code>filter_correct</code>，把 teacher 答错的样本整条剔除，
    所以它<b>从未见过这类题的正确解法</b>，只能停在 base 水平；
    而 RL 只需要一个可验证的判分器，不需要「标准解法」。<br>
    <b>L3</b>：两者皆全错 → 天花板证据，用来说明「并非所有题都能靠方法解决」。
  </div>
</div>

<div class="card">
  <h2>五、统计显著性（配对检验）</h2>
  {sig}
</div>

<div class="grid2">
  <div class="card">{rd}</div>
  <div class="card">{cat_svg}</div>
</div>

<div class="card">
  <h2>六、训练过程</h2>
  <div class="grid2">
    <div>{dist_svg}</div>
    <div>{rl_svg}</div>
  </div>
  <div class="note">
    loss 是否平稳下降，决定蒸馏有没有真的学到东西；
    奖励曲线是否整体上行、格式遵从率是否同时抬升，决定 RL 有没有把行为练出来。
    单看奖励上行而多样性掉到 0，通常是"学会了套模板"而非"学会了做题"。
  </div>
</div>

<div class="card">
  <h2>七、thinking 模式的影响（开 vs 关）</h2>
  <div class="note" style="border-left:3px solid #ba7517">
    <b>⚠ 本表不能横向比较 —— 各角色的 token 预算不同，这是本机算力所迫，不是模型能力差异。</b><br>
    本机角色（base/A/B）的预算是 <b>384 个 token</b>（含思考）。实测它们开思考后，
    推理过程会吃光全部预算、正式答案还没开头就被截断 —— 所以「开」这列正确率显示 0%。<br>
    Ollama 角色（qwen4b / teacher）同样受 384 约束，只有 33%/0% 走到了正式作答。<br>
    云角色（DeepSeek V4 Pro）预算是 <b>1024</b>，思考写完还有余量，所以是 100%。<br>
    <b>结论</b>：本表唯一的有效信息是「<b>预算不足时，思考模式会让输出变差</b>」，
    而不是「谁更会思考」。要做真正的思考能力对比，得把
    <code>think_max_new_tokens</code> 提到 1500 以上 —— 但那意味着本机单题要跑 15 分钟以上，
    在本轮不现实。<br>
    附带一个有意思的观察：原版 Qwen3.5-2B 的思维链是<b>英文</b>的
    （输出形如 <code>Thinking Process: 1. Analyze the Request…</code>），并不是中文。
  </div>
  {think_section(metrics)}
  <div class="note">
    机制：本地模型走 chat template 的 <code>enable_thinking</code>；Ollama 模型走 API 的
    <code>think</code> 参数；云 API 走 <code>thinking</code> 字段
    （<code>{{"type":"disabled"}}</code> 关闭），且推理内容以<b>独立内容块</b>返回，
    比解析 <code>&lt;think&gt;</code> 标签更可靠。<b>判分前必须先剥掉推理块</b>——
    模型在推理里也会写出格式标记，不剥离会把"自言自语"当成正式答案。<br>
    注意本表两列的口径也不同：<b>「关」是完整口径</b>（全部题目、多次采样），
    <b>「开」是小题量探针</b>（开思考后输出会长好几倍，本机 1.32 tok/s 下不控量跑不完）。
  </div>
</div>

<div class="card">
  <h2>八、完整数据表</h2>
  {table_rows()}
</div>

<div class="card">
  <h2>九、口径说明</h2>
  <div class="note">
    <b>格式遵从率</b>：【思路】/【步骤】/【答案】三段齐全且顺序正确，由程序判定。<br>
    <b>任务正确率</b>：把【答案】区抽出来与程序生成的 ground truth 比对，
    支持分数 / 小数 / 带单位 / 等价形式的归一化比较。<br>
    <b>与 teacher 一致性</b>：本模型输出与 9B teacher 输出的文本相似度（空白归一后取比值）。<br>
    <b>输出多样性</b>：同一题多次采样之间的平均差异度；越接近 0 说明输出越趋同。<br>
    <b>通用能力保持</b>：闲聊 / 常识小测的关键词命中率，用来盯灾难性遗忘。<br>
    <b>采样协议</b>：每题用温度 0.7 重复采样 3 次，表中 <span class="pm">±</span>
    是逐题均值的标准差。之所以不报 greedy：温度 0 的解码是<b>确定性的</b>，
    重复跑结果完全一样，取平均没有任何降误差作用。<br>
    <b>多样性</b>直接复用主测的这 3 次采样（同题内两两差异），不需要另跑一轮。<br>
    <b>thinking 模式</b>：两种都测。开思考时先剥掉 <code>&lt;think&gt;…&lt;/think&gt;</code>
    再判分；「开」的条件因输出过长，用小题量探针，故只作趋势参考。
  </div>
</div>

</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    lab_root = lab_path(cfg, "lab_root")
    res = lab_path(cfg, "results")

    mpath = res / "eval_metrics.json"
    if not mpath.exists():
        print(f"✗ 还没有评测结果 {mpath}\n  先跑 30_evaluate.py")
        return 1
    metrics = json.loads(mpath.read_text(encoding="utf-8"))

    dist_log = read_jsonl(lab_path(cfg, "branch_a") / "logs" / "train_distill.jsonl")
    rl_log = read_jsonl(lab_path(cfg, "branch_b") / "logs" / "rl_train.jsonl")

    banner("生成可视化横评报告")
    print(f"  参评模型 {list(metrics['models'])}")
    print(f"  蒸馏训练记录 {len(dist_log)} 条 · RL 训练记录 {len(rl_log)} 条")

    out = res / "横评报告.html"
    ensure_dir(out.parent)
    out.write_text(build_html(cfg, metrics, dist_log, rl_log), encoding="utf-8")
    print(f"  已生成 {out}")
    print(f"  大小 {out.stat().st_size/1024:.1f} KB（自包含，无外部依赖）")

    # 顺带导一份 markdown 方便粘贴
    md = res / "横评报告.md"
    md.write_text(to_markdown(metrics), encoding="utf-8")
    print(f"  已生成 {md}")

    if args.open:
        import subprocess
        subprocess.run(["open", str(out)], check=False)
    return 0


def to_markdown(metrics) -> str:
    roles = [r for r in PALETTE if r in metrics["models"]]
    ms = metrics["models"]
    L = ["# 本地模型实验室 · 蒸馏 vs RL 横评", "",
         f"评测集 {metrics.get('n_eval','—')} 题", "",
         "| 模型 | 格式遵从率 | 任务正确率 | 与teacher一致性 | 输出多样性 | 通用能力 | 均耗时(s) |",
         "|---|---|---|---|---|---|---|"]
    for r in roles:
        m = ms[r]
        gp = "—" if m.get("general_probe") is None else f"{m['general_probe']*100:.1f}%"
        L.append(f"| {PALETTE[r][1]} | {m['format_rate']*100:.1f}% | {m['accuracy']*100:.1f}% | "
                 f"{m['consistency']*100:.1f}% | {m['diversity']*100:.1f}% | {gp} | "
                 f"{m['avg_latency_s']} |")
    L.append("")

    # ---- 分层（主指标）----
    present = [l for l in ("L1", "L2", "L3")
               if any((ms[r].get("by_layer") or {}).get(l) for r in roles)]
    if present:
        L += ["## 分层正确率（主指标）", "",
              "| 模型 | " + " | ".join(LAYER_LABEL[l] for l in present) + " | 总体 | 95% CI |",
              "|---" * (len(present) + 3) + "|"]
        for r in roles:
            m = ms[r]
            cells = []
            for l in present:
                v = (m.get("by_layer") or {}).get(l)
                cells.append(f"{v['accuracy']*100:.1f}%" if v else "—")
            ci = m.get("accuracy_ci") or []
            ci_s = f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]" if len(ci) == 2 else "—"
            L.append(f"| {PALETTE[r][1]} | " + " | ".join(cells) +
                     f" | {m['accuracy']*100:.1f}% | {ci_s} |")
        L.append("")

    # ---- 配对显著性 ----
    pw = metrics.get("pairwise") or {}
    if pw:
        L += ["## 配对显著性（McNemar 精确检验）", "",
              "| 对比 | 仅前者会 | 仅后者会 | 共同题 | p 值 | 结论 |",
              "|---|---|---|---|---|---|"]
        for k in sorted(pw):
            v = pw[k]
            a, b = k.split("_vs_")
            L.append(f"| {a} vs {b} | {v['only_a']} | {v['only_b']} | "
                     f"{v['n_common']} | {v['p_value']} | "
                     f"{'显著' if v['significant'] else '不显著'} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
