#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""32_overview_charts.py —— 为「项目全貌.html」生成内联 SVG 图表

为什么要单独一个脚本，而不是把 SVG 手写进 HTML：
  1. 数据来自 `shared/results/eval_metrics.json`，脚本生成 = **数字永远不会手抄错**
  2. 图表格式（坐标、颜色、标签错开）集中在一处，改一次全生效
  3. 重跑安全：会先把原 HTML 备份到 logs/ 下

产出两张图：
  · grouped-bars  分组柱状图：6 个角色 × 3 层难度
  · layer-trend   折线图：每个角色从 L1 → L2 → L3 的走势（最能看出 A1 断崖）

用法：
    python scripts/32_overview_charts.py            # 生成并就地更新 HTML
    python scripts/32_overview_charts.py --dry-run  # 只打印统计，不写文件

设计约束：
  · **全部内联**，不引 CDN —— 断网也能打开（这是本项目的既定纪律）
  · 颜色与字号尽量用页面已有的 CSS 变量，深浅色模式自动适配
  · 每个图形元素都带 <title>，鼠标悬停能看原始数值
"""
import json
import os
import re
import shutil
import sys
import datetime

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(LAB, "shared/results/eval_metrics.json")
HTML = os.path.join(LAB, "项目全貌.html")

# 角色显示顺序 + 通俗名 + 颜色（与页面里的说明一致）
ROLES = [
    ("base",     "原版",        "#94a3b8"),
    ("A1",       "A1 蒸馏",     "#7c3aed"),
    ("B",        "B 强化",      "#16a34a"),
    ("C",        "C 混合",      "#d97706"),
    ("teacher",  "老师",        "#dc2626"),
    ("dsv4pro",  "云端",        "#2563eb"),
]

LAYERS = [("L1", "L1 基础题", "#22c55e"),
          ("L2", "L2 老师不会", "#dc2626"),
          ("L3", "L3 都不会", "#8b5cf6")]


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def load():
    m = json.load(open(METRICS, encoding="utf-8"))
    data = {}
    for key, _, _ in ROLES:
        bl = m["models"][key]["by_layer"]
        data[key] = {l: bl[l]["accuracy"] * 100 for l, _, _ in LAYERS}
    return data


# ---------------------------------------------------------------- 柱状图
def grouped_bars(data, width=760, height=380):
    """6 个角色 × 3 层难度的分组柱状图。"""
    pl, pr, pt, pb = 54, 18, 46, 62
    iw, ih = width - pl - pr, height - pt - pb
    gw = iw / len(ROLES)              # 组宽
    bw = 26                            # 单柱宽
    gap = 7
    group_inner = bw * 3 + gap * 2     # 组内总宽
    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
         f'aria-label="各角色在三个难度层上的正确率对比" '
         f'style="display:block;margin:6px 0 2px">']

    # 图例
    lx = pl
    for _, label, color in LAYERS:
        p.append(f'<rect x="{lx}" y="14" width="13" height="13" rx="3" fill="{color}"/>')
        p.append(f'<text x="{lx+19}" y="25" font-size="13" fill="var(--muted)">{esc(label)}</text>')
        lx += 22 + len(label) * 13 + 26

    # 网格 + y 轴刻度
    for v in (0, 25, 50, 75, 100):
        y = pt + ih * (1 - v / 100)
        p.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+iw}" y2="{y:.1f}" '
                 f'stroke="var(--line)" stroke-width="1" '
                 f'{"" if v else ""}/>')
        p.append(f'<text x="{pl-9}" y="{y+4:.1f}" font-size="11" fill="var(--muted)" '
                 f'text-anchor="end">{v}%</text>')

    # 柱子
    for gi, (key, disp, _) in enumerate(ROLES):
        gx = pl + gi * gw + (gw - group_inner) / 2
        for li, (layer, llabel, lcolor) in enumerate(LAYERS):
            v = data[key][layer]
            h = ih * v / 100
            x = gx + li * (bw + gap)
            y = pt + ih - h
            p.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{max(h,1):.1f}" '
                f'rx="3" fill="{lcolor}" opacity="0.92">'
                f'<title>{esc(disp)} · {esc(llabel)}：{v:.1f}%</title></rect>')
            # 数值标签（小于 6% 的柱子太矮，标签挪到柱子上方外侧）
            if v >= 6:
                ty = y - 4
                p.append(f'<text x="{x+bw/2:.1f}" y="{ty:.1f}" font-size="10.5" '
                         f'fill="var(--text)" text-anchor="middle">{v:.0f}</text>')
            else:
                p.append(f'<text x="{x+bw/2:.1f}" y="{pt+ih-6:.1f}" font-size="10" '
                         f'fill="var(--muted)" text-anchor="middle">{v:.0f}</text>')
        # 组标签
        p.append(f'<text x="{pl+gi*gw+gw/2:.1f}" y="{pt+ih+20:.1f}" font-size="12.5" '
                 f'fill="var(--text)" text-anchor="middle">{esc(disp)}</text>')

    # 轴线
    p.append(f'<line x1="{pl}" y1="{pt+ih}" x2="{pl+iw}" y2="{pt+ih}" '
             f'stroke="var(--line)" stroke-width="1.5"/>')
    p.append('</svg>')
    return "\n".join(p)


# ---------------------------------------------------------------- 折线图
def layer_trend(data, width=760, height=422):
    """6 个角色从 L1 → L2 → L3 的走势折线图。

    底部留白（pb）特意给得大：值为 0% 的点，其数值标签要往下沉才能错开，
    留白不够就会压到横轴标题上（这是实测抓到的重叠）。"""
    pl, pr, pt, pb = 54, 96, 44, 96
    iw, ih = width - pl - pr, height - pt - pb
    n = len(LAYERS)

    def X(i):
        return pl + (iw / (n - 1)) * i if n > 1 else pl + iw / 2

    def Y(v):
        return pt + ih * (1 - v / 100)

    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
         f'aria-label="各角色从基础题到难题的走势" style="display:block;margin:6px 0 2px">']

    # 网格
    for v in (0, 25, 50, 75, 100):
        y = Y(v)
        p.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+iw}" y2="{y:.1f}" stroke="var(--line)"/>')
        p.append(f'<text x="{pl-9}" y="{y+4:.1f}" font-size="11" fill="var(--muted)" '
                 f'text-anchor="end">{v}%</text>')
    # 横轴刻度标签（放在最下方，远离任何数值标签）
    for i, (_, llabel, _) in enumerate(LAYERS):
        p.append(f'<text x="{X(i):.1f}" y="{pt+ih+50:.1f}" font-size="12.5" '
                 f'fill="var(--text)" text-anchor="middle">{esc(llabel)}</text>')
    p.append(f'<line x1="{pl}" y1="{pt+ih}" x2="{pl+iw}" y2="{pt+ih}" '
             f'stroke="var(--line)" stroke-width="1.5"/>')

    # 每个 x 位置上，把过近的标签上下错开。
    #
    # 【坑 1】判重必须用「标签的实际落点」，不能只比较「数值」——
    #   曾经写成「值相近 且 偏移相同」才算冲突，结果：
    #   70% 占了 dy=13，把 66.7% 挤到 dy=26，而 53.3% 仍用 dy=-10，
    #   两者在屏幕上只差 1.8px，直接压在一起。
    #
    # 【坑 2】换算的符号容易搞反：纵轴是「值越大、y 越小」，
    #   所以像素往下移（dy>0）在数值空间里是**减**：
    #       落点等效值 = v - dy / 绘图高度 * 100
    #   写成 `v + ...` 会让检查全部失效（实测重叠反而从 1 对涨到 5 对）。
    offsets = {}
    for i in range(n):
        vals = sorted(((data[k][LAYERS[i][0]], k) for k, _, _ in ROLES), reverse=True)
        placed = []          # 已占用位置（统一换算到"数值空间"比较）
        for v, k in vals:
            cand = [-10, 13, 26]
            dy = cand[-1]
            pos = v - dy / ih * 100
            for c in cand:
                tp = v - c / ih * 100
                if all(abs(tp - u) > 6.5 for u in placed):
                    dy, pos = c, tp
                    break
            placed.append(pos)
            offsets[(k, i)] = dy

    # 折线（先画线，再画点，保证点在线上面）
    for key, disp, color in ROLES:
        pts = " ".join(f"{X(i):.1f},{Y(data[key][LAYERS[i][0]]):.1f}" for i in range(n))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" '
                 f'opacity="0.95"/>')
    for key, disp, color in ROLES:
        for i in range(n):
            v = data[key][LAYERS[i][0]]
            cx, cy = X(i), Y(v)
            p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.6" fill="{color}" '
                     f'stroke="var(--card)" stroke-width="1.8">'
                     f'<title>{esc(disp)} · {esc(LAYERS[i][1])}：{v:.1f}%</title></circle>')
            p.append(f'<text x="{cx+9:.1f}" y="{cy+offsets[(key,i)]:.1f}" font-size="10.5" '
                     f'font-weight="600" fill="{color}" '
                     f'style="paint-order:stroke;stroke:var(--card);stroke-width:3px">'
                     f'{v:.0f}%</text>')

    # 右侧图例
    ly = pt + 6
    for key, disp, color in ROLES:
        p.append(f'<line x1="{pl+iw+14}" y1="{ly+6}" x2="{pl+iw+38}" y2="{ly+6}" '
                 f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        p.append(f'<text x="{pl+iw+44}" y="{ly+10}" font-size="12" fill="var(--text)">'
                 f'{esc(disp)}</text>')
        ly += 24

    p.append('</svg>')
    return "\n".join(p)


# ---------------------------------------------------------------- 主流程
def main():
    dry = "--dry-run" in sys.argv
    data = load()

    print("═" * 62)
    print("  数据核对（来自 eval_metrics.json，不是手抄）")
    print("═" * 62)
    print(f"  {'角色':<10}" + "".join(f"{l:>10}" for l, _, _ in LAYERS))
    for key, disp, _ in ROLES:
        print(f"  {disp:<10}" + "".join(f"{data[key][l]:>9.1f}%" for l, _, _ in LAYERS))

    if dry:
        print("\n  [dry-run] 不写文件")
        return 0

    if not os.path.exists(HTML):
        print(f"✗ 找不到 {HTML}")
        return 1

    src = open(HTML, encoding="utf-8").read()

    # 两种情形都要能处理，脚本才可重复运行：
    #   首次 → 页面里是空的占位符 <!-- CHART:name -->
    #   再次 → 页面里是上次生成的整块 <!-- CHART:BEGIN name -->…<!-- CHART:END name -->
    def render_block(name, svg):
        return f"<!-- CHART:BEGIN {name} -->\n{svg}\n<!-- CHART:END {name} -->"

    charts = [("grouped-bars", grouped_bars(data)),
              ("layer-trend", layer_trend(data))]

    new, replaced = src, 0
    for name, svg in charts:
        block = render_block(name, svg)
        if f"<!-- CHART:BEGIN {name} -->" in new:
            new = re.sub(rf"<!-- CHART:BEGIN {name} -->.*?<!-- CHART:END {name} -->",
                         lambda _m, b=block: b, new, flags=re.S)
            replaced += 1
        elif f"<!-- CHART:{name} -->" in new:
            new = new.replace(f"<!-- CHART:{name} -->", block)
            replaced += 1

    if replaced == 0:
        print("✗ HTML 里既没有占位符、也没有已生成的图表块")
        return 1

    # 备份（按项目约定：改动前留底）
    bak_dir = os.path.join(LAB, "logs/archive_prelaunch")
    os.makedirs(bak_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    bak = os.path.join(bak_dir, f"项目全貌_before_charts_{stamp}.html")
    shutil.copyfile(HTML, bak)
    print(f"\n  已留底 → {os.path.relpath(bak, LAB)}")

    open(HTML, "w", encoding="utf-8").write(new)

    print(f"  已更新 {replaced} 张图表")
    print(f"  文件 {len(src.encode('utf-8'))} → {len(new.encode('utf-8'))} 字节")
    return 0


if __name__ == "__main__":
    sys.exit(main())
