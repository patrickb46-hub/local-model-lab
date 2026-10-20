#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_site.py —— 把 Local-Model-Lab 打包成「中英双语」可在线浏览的静态站点。

输出：<项目>/site/         （只有这个目录会被发布到线上）
英文源：<项目>/i18n/       （翻译产物，不参与发布）
原则：只发布「代码 / 文档 / 实验报告」，绝不包含模型权重。

用法：python3 build_site.py   （需额外装 markdown；不是训练用的那个环境）
"""

import html
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

import markdown

ROOT = Path(__file__).resolve().parent
I18N = ROOT / "i18n"
OUT = ROOT / "site"

# --------------------------------------------------------------------------- #
# 页面注册表：key -> (中文文件名, 英文文件名)
# --------------------------------------------------------------------------- #
PAGE_FILES = {
    "index":    ("index.html",    "index.en.html"),
    "overview": ("overview.html", "overview.en.html"),
    "report":   ("report.html",   "report.en.html"),
    "docs":     ("docs.html",     "docs.en.html"),
    "source":   ("source.html",   "source.en.html"),
    "theory":   ("theory.html",   "theory.en.html"),
    "plan":     ("plan.html",     "plan.en.html"),
    "showcase": ("showcase.html", "showcase.en.html"),
    "lessons":  ("lessons.html",  "lessons.en.html"),
    "pitfalls": ("pitfalls.html", "pitfalls.en.html"),
    "readme":   ("readme.html",   "readme.en.html"),
}

# 导航栏项目（顺序即显示顺序）
NAV_ORDER = ["home", "overview", "report", "docs", "source"]

# 每个页面高亮哪一项导航
ACTIVE_OF = {
    "index": "home", "overview": "overview", "report": "report",
    "docs": "docs", "theory": "docs", "plan": "docs", "showcase": "docs",
    "lessons": "docs", "pitfalls": "docs", "readme": "docs",
    "source": "source",
}

NAV_LABEL = {
    "zh": {"home": "首页", "overview": "项目全貌", "report": "评测报告",
           "docs": "文档笔记", "source": "源码"},
    "en": {"home": "Home", "overview": "Overview", "report": "Report",
           "docs": "Docs", "source": "Source"},
}

NAV_HREF = {
    "zh": {"home": "index.html", "overview": "overview.html", "report": "report.html",
           "docs": "docs.html", "source": "source.html"},
    "en": {"home": "index.en.html", "overview": "overview.en.html",
           "report": "report.en.html", "docs": "docs.en.html",
           "source": "source.en.html"},
}

LANG_BTN = {"zh": "EN", "en": "中文"}
LANG_OTHER = {"zh": "en", "en": "zh"}

# --------------------------------------------------------------------------- #
# 导航样式与结构
# --------------------------------------------------------------------------- #
NAV_CSS = """<style id="sn-nav-css">
.sn-nav{position:sticky;top:0;z-index:999;background:rgba(255,255,255,.93);-webkit-backdrop-filter:saturate(180%) blur(14px);backdrop-filter:saturate(180%) blur(14px);border-bottom:1px solid #e6ebf2;font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.sn-nav .sn-in{max-width:960px;margin:0 auto;padding:0 24px;height:54px;display:flex;align-items:center;gap:2px;overflow-x:auto;scrollbar-width:none}
.sn-nav .sn-in::-webkit-scrollbar{display:none}
.sn-nav .sn-brand{font-weight:700;font-size:15px;color:#0f172a;text-decoration:none;margin-right:auto;letter-spacing:-.3px;white-space:nowrap;padding-right:16px}
.sn-nav .sn-brand b{color:#2563eb}
.sn-nav a.sn-link{color:#5b6b80;text-decoration:none;padding:7px 12px;border-radius:9px;white-space:nowrap;font-weight:500;transition:.15s}
.sn-nav a.sn-link:hover{background:#f1f5f9;color:#0f172a}
.sn-nav a.sn-link.sn-on{background:#eff6ff;color:#2563eb;font-weight:600}
.sn-nav a.sn-lang{color:#2563eb;border:1px solid #dbe3ee;padding:5px 12px;border-radius:8px;font-weight:600;font-size:12.5px;margin-left:10px;text-decoration:none;white-space:nowrap;transition:.15s}
.sn-nav a.sn-lang:hover{background:#eff6ff;border-color:#2563eb}
@media (prefers-color-scheme:dark){
.sn-nav{background:rgba(11,18,32,.93);border-bottom-color:#243149}
.sn-nav .sn-brand{color:#e5edf8}
.sn-nav .sn-brand b{color:#60a5fa}
.sn-nav a.sn-link{color:#93a3b8}
.sn-nav a.sn-link:hover{background:#1b273c;color:#e5edf8}
.sn-nav a.sn-link.sn-on{background:#16274a;color:#60a5fa}
.sn-nav a.sn-lang{color:#60a5fa;border-color:#2b3d5c}
.sn-nav a.sn-lang:hover{background:#16274a;border-color:#60a5fa}
}
</style>
"""


def nav_html(page_key, lang):
    """生成某一页的导航栏（含语言切换按钮）。"""
    active = ACTIVE_OF[page_key]
    links = []
    for key in NAV_ORDER:
        cls = "sn-link sn-on" if key == active else "sn-link"
        links.append('<a class="%s" href="%s">%s</a>'
                     % (cls, NAV_HREF[lang][key], NAV_LABEL[lang][key]))
    # 语言切换：指向同一页面的另一语言版本
    other = LANG_OTHER[lang]
    other_href = PAGE_FILES[page_key][1 if other == "en" else 0]
    switch = '<a class="sn-lang" href="%s">%s</a>' % (other_href, LANG_BTN[lang])
    return (NAV_CSS + '<nav class="sn-nav"><div class="sn-in">'
            '<a class="sn-brand" href="%s">Local<b>Model</b>Lab</a>'
            % NAV_HREF[lang]["home"]
            + "".join(links) + switch + "</div></nav>\n")


# --------------------------------------------------------------------------- #
# 生成页样式
# --------------------------------------------------------------------------- #
PAGE_CSS = """<style>
:root{--bg:#f6f8fb;--card:#fff;--text:#1e293b;--muted:#64748b;--line:#e2e8f0;--blue:#2563eb;--blue-bg:#eff6ff;--green:#16a34a;--green-bg:#f0fdf4;--red:#dc2626;--red-bg:#fef2f2;--amber:#d97706;--amber-bg:#fffbeb;--purple:#7c3aed;--purple-bg:#faf5ff;--shadow:0 1px 3px rgba(15,23,42,.06),0 8px 24px rgba(15,23,42,.06)}
@media (prefers-color-scheme:dark){:root{--bg:#0b1220;--card:#141e30;--text:#e5edf8;--muted:#94a3b8;--line:#25324a;--blue:#60a5fa;--blue-bg:#152447;--green:#4ade80;--green-bg:#12291d;--red:#f87171;--red-bg:#2a1517;--amber:#fbbf24;--amber-bg:#2a2110;--purple:#a78bfa;--purple-bg:#221a3a;--shadow:0 1px 3px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3)}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:960px;margin:0 auto;padding:0 24px 72px}
h1{font-size:30px;line-height:1.35;letter-spacing:-.6px;margin:38px 0 10px}
h2{font-size:22px;margin:46px 0 16px;padding-bottom:10px;border-bottom:2px solid var(--line);letter-spacing:-.3px}
h3{font-size:17px;margin:28px 0 10px}
p{margin:0 0 14px}
a{color:var(--blue)}
.lead{font-size:17px;color:var(--muted);margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 24px;box-shadow:var(--shadow);margin:18px 0}
table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14.5px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line)}
th{background:var(--blue-bg);font-weight:600}
tr:last-child td{border-bottom:0}
code{background:var(--blue-bg);padding:1.5px 6px;border-radius:5px;font-size:13px;font-family:ui-monospace,Menlo,Consolas,monospace}
pre{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;overflow:auto;font-size:13px;line-height:1.62}
pre code{background:none;padding:0;font-size:13px}
blockquote{margin:16px 0;padding:12px 18px;background:var(--blue-bg);border-left:4px solid var(--blue);border-radius:8px}
blockquote p:last-child{margin:0}
hr{border:0;border-top:1px solid var(--line);margin:36px 0}
ul,ol{padding-left:24px}
li{margin-bottom:6px}
footer{margin-top:60px;padding-top:22px;border-top:1px solid var(--line);font-size:13px;color:var(--muted);line-height:1.8}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(258px,1fr));gap:16px;margin:24px 0}
.gcard{display:block;text-decoration:none;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;box-shadow:var(--shadow);transition:.18s}
.gcard:hover{transform:translateY(-2px);border-color:var(--blue)}
.gcard h3{margin:0 0 8px;font-size:16px}
.gcard p{margin:0;font-size:13.5px;color:var(--muted);line-height:1.6}
.tag{display:inline-block;font-size:11.5px;font-weight:600;color:var(--blue);background:var(--blue-bg);border-radius:20px;padding:2px 10px;margin-bottom:10px}
.hero{background:linear-gradient(135deg,#1e40af,#3b82f6 55%,#60a5fa);color:#fff;border-radius:16px;padding:34px 32px;margin:26px 0 28px;box-shadow:0 10px 30px rgba(37,99,235,.22)}
.hero h1{margin:0 0 12px;font-size:30px;color:#fff}
.hero p{margin:0;opacity:.95;font-size:16px;line-height:1.7}
.stat{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:14px;margin:22px 0}
.stat div{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 12px;text-align:center;box-shadow:var(--shadow)}
.stat b{display:block;font-size:25px;color:var(--blue);letter-spacing:-1px;line-height:1.3}
.stat span{font-size:12.5px;color:var(--muted);line-height:1.5;display:block}
.hl{background:var(--green-bg);border:1px solid var(--green);border-left:5px solid var(--green);border-radius:10px;padding:16px 20px;margin:20px 0}
.hl.warn{background:var(--amber-bg);border-color:var(--amber);border-left-color:var(--amber)}
.hl p:last-child{margin:0}
details.src{border:1px solid var(--line);border-radius:10px;margin:10px 0;background:var(--card);overflow:hidden}
details.src>summary{cursor:pointer;padding:11px 16px;font-size:13.5px;font-weight:600;font-family:ui-monospace,Menlo,Consolas,monospace;list-style:none}
details.src>summary::-webkit-details-marker{display:none}
details.src>summary::before{content:"\\25B8  ";color:var(--blue)}
details.src[open]>summary::before{content:"\\25BE  "}
details.src>summary:hover{background:var(--blue-bg)}
details.src pre{margin:0;border:0;border-radius:0;border-top:1px solid var(--line);max-height:520px}
details.src .m2{float:right;font-weight:400;color:var(--muted);font-size:12.5px;font-family:-apple-system,sans-serif}
.filetree{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 22px;overflow:auto;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.75;white-space:pre;box-shadow:var(--shadow)}
</style>
"""

FOOTER = {
    "zh": ("本站由项目文档自动生成 · <b>本地模型实验室 Local Model Lab</b> · 2026 年 10 月<br>"
           "模型权重（约 22 GB）未包含在内，本站只展示代码、文档与实验报告 —— 全部结果在一台 "
           "2018 款 Intel MacBook Pro 上纯 CPU 跑出。"),
    "en": ("Generated automatically from the project's own documentation · "
           "<b>Local Model Lab</b> · October 2026<br>"
           "Model weights (~22 GB) are not included — this site carries only the code, docs and "
           "evaluation reports. Everything was run CPU-only on a 2018 Intel MacBook Pro."),
}

PAGE_TMPL = """<!DOCTYPE html>
<html lang="%%LANG%%">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
%%CSS%%
</head>
<body>
%%NAV%%
<div class="wrap">
%%BODY%%
<footer>%%FOOTER%%</footer>
</div>
</body>
</html>
"""


def build_page(page_key, lang, title, body, css=PAGE_CSS):
    return (PAGE_TMPL
            .replace("%%LANG%%", "zh-CN" if lang == "zh" else "en")
            .replace("%%TITLE%%", html.escape(title))
            .replace("%%CSS%%", css)
            .replace("%%NAV%%", nav_html(page_key, lang))
            .replace("%%BODY%%", body)
            .replace("%%FOOTER%%", FOOTER[lang]))


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def clean(text):
    """路径脱敏：公开分享时不暴露本机用户名（动态取家目录，不写死用户名）。"""
    text = text.replace(str(ROOT), "~/Desktop/AI/Local-Model-Lab")
    return text.replace(str(Path.home()), "~")


def esc(t):
    return html.escape(t, quote=False)


def md2html(text):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc",
                                       "sane_lists", "attr_list"])
    return md.convert(text)


def inject_nav(raw_html, page_key, lang):
    """把导航栏注入已有的自包含 HTML 页面（保留其原有样式），并统一 lang 属性。"""
    raw_html = re.sub(r"<html[^>]*>",
                      '<html lang="%s">' % ("zh-CN" if lang == "zh" else "en"),
                      raw_html, count=1)
    return re.sub(r"<body([^>]*)>",
                  lambda m: "<body%s>\n%s" % (m.group(1), nav_html(page_key, lang)),
                  raw_html, count=1)


def describe(path):
    """从文件开头的注释 / docstring 里取一句简介（源码页用，仅中文版）。"""
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    for ln in txt.splitlines()[:30]:
        s = ln.strip()
        if not s or s.startswith("#!"):
            continue
        if s.startswith("#"):
            s = s.lstrip("#").strip(" =-").strip()
            if s:
                return s[:100]
        elif s.startswith(('"""', "'''")):
            s = s.strip("\"'").strip()
            if s:
                return s[:100]
        else:
            return ""
    return ""


# --------------------------------------------------------------------------- #
# 文档注册表（中英双语）
# --------------------------------------------------------------------------- #
DOCS = [
    {
        "key": "theory",
        "zh_src": ROOT / "docs/原理笔记.md",
        "en_src": I18N / "theory.en.md",
        "zh_title": "原理笔记",
        "en_title": "How It Works (Technical Notes)",
        "zh_desc": "蒸馏与强化学习到底在做什么，数学上有什么区别。",
        "en_desc": "What distillation and reinforcement learning actually do, and how they differ.",
    },
    {
        "key": "plan",
        "zh_src": ROOT / "完整计划.md",
        "en_src": I18N / "plan.en.md",
        "zh_title": "完整计划",
        "en_title": "The Full Plan",
        "zh_desc": "项目启动时写下的完整设计：目标、路线、验收标准与取舍。",
        "en_desc": "The complete design written at kickoff: goals, route, acceptance criteria and trade-offs.",
    },
    {
        "key": "showcase",
        "zh_src": ROOT / "成果展示.md",
        "en_src": I18N / "showcase.en.md",
        "zh_title": "成果展示",
        "en_title": "Results Showcase",
        "zh_desc": "第一轮跑完后的成果记录：模型跑起来什么样、评测怎么读。",
        "en_desc": "What the first full run produced, and how to read the numbers.",
    },
    {
        "key": "lessons",
        "zh_src": ROOT / "docs/经验教训总纲.md",
        "en_src": I18N / "lessons.en.md",
        "zh_title": "经验教训总纲",
        "en_title": "Key Lessons Learned",
        "zh_desc": "从整个项目里提炼出的通用经验，包括 token 预算、评测口径这些容易踩的坑。",
        "en_desc": "General lessons drawn from the whole project, including token budgets and evaluation pitfalls.",
    },
    {
        "key": "pitfalls",
        "zh_src": ROOT / "docs/踩坑记录.md",
        "en_src": I18N / "pitfalls.en.md",
        "zh_title": "踩坑记录",
        "en_title": "Pitfalls Log",
        "zh_desc": "24 个真实踩过的坑：从 Intel Mac 的 torch 死局到推理模型的思考块，附爬出来的办法。",
        "en_desc": "24 real pitfalls — from the Intel Mac PyTorch dead-end to the reasoning model's thinking blocks — and how each was escaped.",
    },
    {
        "key": "readme",
        "zh_src": ROOT / "README.md",
        "en_src": I18N / "readme.en.md",
        "zh_title": "README · 技术说明",
        "en_title": "README · Technical Notes",
        "zh_desc": "给技术读者的完整说明书：运行环境、目录结构、脚本用法、复现步骤。",
        "en_desc": "The full spec for engineers: environment, directory layout, script usage and reproduction steps.",
    },
]

MISSING = []

# --------------------------------------------------------------------------- #
# 站内链接重写
# README 等文档里有很多「指向项目文件」的相对链接（如 [完整计划.md](完整计划.md)）。
# 在网站上这些 .md 文件并不存在，直接点会 404。这里统一改指到站点内的对应页面。
# --------------------------------------------------------------------------- #
LINK_MAP = {
    "README.md": "readme",
    "完整计划.md": "plan",
    "成果展示.md": "showcase",
    "项目全貌.html": "overview",
    "项目全貌.md": "overview",
    "docs/原理笔记.md": "theory",
    "docs/经验教训总纲.md": "lessons",
    "docs/踩坑记录.md": "pitfalls",
    # 有的文档写的是不带目录前缀的裸文件名，两种都要认
    "原理笔记.md": "theory",
    "经验教训总纲.md": "lessons",
    "踩坑记录.md": "pitfalls",
    "shared/results/横评报告.html": "report",
    "横评报告.html": "report",
    "横评报告.md": "report",
}

TITLE_OF = {
    "readme": ("README · 技术说明", "README · Technical Notes"),
    "plan": ("完整计划", "The Full Plan"),
    "showcase": ("成果展示", "Results Showcase"),
    "theory": ("原理笔记", "How It Works (Technical Notes)"),
    "lessons": ("经验教训总纲", "Key Lessons Learned"),
    "pitfalls": ("踩坑记录", "Pitfalls Log"),
    "overview": ("项目全貌", "Project Overview"),
    "report": ("评测报告", "Evaluation Report"),
}


def rewrite_links(html_text, lang):
    """把指向项目文件的链接改写成站点内的对应页面（文字也换成页面标题）。"""
    def repl(m):
        base = unquote(m.group(1)).split("#")[0].lstrip("./")
        # 先按完整相对路径查，再按裸文件名兜底
        key = LINK_MAP.get(base) or LINK_MAP.get(base.split("/")[-1])
        if not key:
            return m.group(0)
        target = PAGE_FILES[key][1 if lang == "en" else 0]
        text = TITLE_OF[key][1 if lang == "en" else 0]
        return '<a href="%s">%s</a>' % (target, text)
    # 注意：链接文字里可能含标签（如 [`完整计划.md`](完整计划.md) 会渲染成
    # <a href=".."><code>..</code></a>），所以不能用 [^<]*，要用非贪婪匹配。
    return re.sub(r'<a href="([^"]+)">(.*?)</a>', repl, html_text, flags=re.S)


def read_src(path):
    if not path.exists():
        MISSING.append(str(path.relative_to(ROOT)))
        return None
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 首页
# --------------------------------------------------------------------------- #
INDEX_ZH = """
<div class="hero">
  <h1>一个 AI 学生，能不能超过它的老师？</h1>
  <p>本地模型实验室 · 蒸馏 vs 强化学习 对照实验<br>
  2026 年 10 月 ｜ 全部流程在一台 2018 款 Intel MacBook Pro 上纯 CPU 跑完</p>
</div>

<div class="stat">
  <div><b>16.7%</b><span>抄老师解题过程<br>（老师不会的题）</span></div>
  <div><b>56.2%</b><span>自己刷题<br>（强化学习）</span></div>
  <div><b>39.6%</b><span>什么都不学<br>（原版）</span></div>
  <div><b>0%</b><span>老师本人<br>（同样这些题）</span></div>
</div>

<div class="hl">
  <p><b>结论：能。</b>在「老师自己也不会做」的题目上，只抄老师解题过程的那条路，
  <b>比完全不学还差</b>（16.7% 对 39.6%）—— 它连老师错的地方一起抄了；
  而靠自己刷题的那条做到了 <b>56.2%</b>，<b>越过了老师</b>。</p>
</div>

<h2>这是一件什么事</h2>
<p>有一个<b>小学生</b>（一个小型 AI 模型），要练数学。他旁边有一个<b>大学生</b>（一个更大的 AI 模型），水平比他高。
  这个实验让小学生用两种完全不同的方式去练，然后考同一张卷子，看哪种方式真的有用。</p>

<table>
  <tr><th>练法</th><th>打个比方</th><th>老师不会的题上得分</th></tr>
  <tr><td>蒸馏</td><td>把大学生怎么一步步解题，全抄一遍</td><td><b>16.7%</b> ← 比不学还差</td></tr>
  <tr><td>强化学习</td><td>没人教，自己做题、自己对答案、反复几十轮</td><td><b>56.2%</b> ← 超过老师</td></tr>
  <tr><td>什么都不学</td><td>原版小学生</td><td>39.6%</td></tr>
</table>

<div class="hl warn">
  <p><b>为什么只看「老师不会的题」？</b>
  因为只看「老师会的题」，抄作业的人一定赢 —— 他抄的就是那些。
  只有在<b>老师自己也不会的题</b>上，才能看出一个人是<b>真的会思考，还是只会模仿</b>。
  这些题由程序自动生成、答案唯一，对错百分之百客观，不存在评分争议。</p>
</div>

<h2>从哪里开始看</h2>
<div class="grid">
  <a class="gcard" href="overview.html">
    <span class="tag">推荐先看</span>
    <h3>项目全貌</h3>
    <p>用打比方的方式讲清楚这个项目在做什么、结论是什么。写给非技术读者，带柱状图与折线图。</p>
  </a>
  <a class="gcard" href="report.html">
    <h3>评测报告</h3>
    <p>完整的实验数据：6 个角色 × 52 道题 × 3 次采样，含分层对比、统计检验与 thinking 模式的影响。</p>
  </a>
  <a class="gcard" href="docs.html">
    <h3>文档笔记</h3>
    <p>完整计划、原理笔记、24 个真实踩过的坑、经验教训总纲。做实验的人最该看的部分。</p>
  </a>
  <a class="gcard" href="source.html">
    <h3>源码</h3>
    <p>全部实验脚本、配置文件与数据样例，点开即可阅读。</p>
  </a>
</div>

<h2>项目里有什么</h2>
<div class="filetree">Local-Model-Lab/
│
├── base/                  ← 原版小学生（Qwen3.5-2B，只读，永不修改）
├── branch-A-distill/      ← 走「抄作业」这条路的全部东西（含训练成果 A1）
├── branch-B-rl/           ← 走「自己刷题」这条路的全部东西（含训练成果 B / C）
│
├── shared/                ← 两条路共用的东西
│   ├── results/           ★ 评测报告与全部原始数据
│   └── bench_stratified.jsonl  ★ 冻结的考卷（52 题）
│
├── scripts/               ← 所有程序（出题 / 训练 / 评测 / 出报告）
├── docs/                  ← 笔记（原理、踩坑、经验教训）
├── configs/               ← 所有参数设置
│
├── README.md              ← 给技术读者的说明书
├── 完整计划.md            ← 最初的完整计划
├── 成果展示.md            ← 第一轮的成果
└── 项目全貌.html / .md    ← 给非技术读者的那一份</div>

<p style="font-size:13.5px;color:var(--muted);margin-top:16px">
  树上标注 ★ 的是最值得看的几处。<b>模型权重文件（.safetensors，约 22 GB）没有包含在这个网站里</b>
  —— 它们既不适合在线传输，也不是理解这个实验所必需的。</p>

<h2>怎么做到的</h2>
<p>从动手到出结论，一共约 40 小时机器时间：出题与筛选（7 小时）→ 锁定考卷 → 训练三条支线 →
统一考试（936 次作答）→ 自动出报告。中间还出过一次事故：第一次考试时程序疏忽，
考卷里只混进了基础题，最难的两层一道都没考，于是补考了一次才有了完整结果 —— 这段记录在
<a href="pitfalls.html">踩坑记录</a>里。</p>
"""

INDEX_EN = """
<div class="hero">
  <h1>Can a small AI student surpass its teacher?</h1>
  <p>Local Model Lab · A distillation vs. reinforcement learning controlled experiment<br>
  October 2026 ｜ Every step run CPU-only on a 2018 Intel MacBook Pro</p>
</div>

<div class="stat">
  <div><b>16.7%</b><span>Copying the teacher<br>(distillation, on teacher-fails problems)</span></div>
  <div><b>56.2%</b><span>Practising solo<br>(reinforcement learning)</span></div>
  <div><b>39.6%</b><span>Not training at all<br>(the base model)</span></div>
  <div><b>0%</b><span>The teacher itself<br>(on those same problems)</span></div>
</div>

<div class="hl">
  <p><b>Answer: yes.</b> On problems the teacher itself cannot solve, the branch that merely copied
  the teacher's reasoning did <b>worse than not training at all</b> (16.7% vs. 39.6%) — it copied the
  teacher's mistakes along with everything else. The branch that practiced on its own reached
  <b>56.2%</b>, <b>surpassing its teacher</b>.</p>
</div>

<h2>What this is</h2>
<p>A <b>small student</b> (a small AI model) is practicing maths, with a <b>big student</b>
(a larger AI model) sitting next to it. The experiment trains the small one in two fundamentally
different ways, then sets both the same exam, to see which way actually works.</p>

<table>
  <tr><th>Training method</th><th>In plain terms</th><th>Score on teacher-fails problems</th></tr>
  <tr><td>Distillation</td><td>Copy the big student's step-by-step reasoning, in full</td><td><b>16.7%</b> ← worse than not training</td></tr>
  <tr><td>Reinforcement learning</td><td>Nobody teaches it. It solves problems, checks its own answers, repeats for dozens of rounds</td><td><b>56.2%</b> ← beats the teacher</td></tr>
  <tr><td>No training</td><td>The original small student</td><td>39.6%</td></tr>
</table>

<div class="hl warn">
  <p><b>Why look only at problems the teacher cannot solve?</b>
  Because on problems the teacher <i>can</i> solve, whoever copied the homework always wins — those are
  exactly the problems they copied. Only on problems the teacher itself gets wrong can you tell
  whether someone <b>can really think, or can only imitate</b>. These problems are generated
  programmatically with unique answers, so grading is 100% objective — there is no room for
  disagreement about the marking scheme.</p>
</div>

<h2>Where to start</h2>
<div class="grid">
  <a class="gcard" href="overview.en.html">
    <span class="tag">Start here</span>
    <h3>Overview</h3>
    <p>Explains in plain language what this project does and what it found. Written for non-technical
    readers, with bar and line charts.</p>
  </a>
  <a class="gcard" href="report.en.html">
    <h3>Evaluation Report</h3>
    <p>The full experimental data: 6 roles × 52 problems × 3 samples, with tier-by-tier comparison,
    statistical tests and the effect of thinking mode.</p>
  </a>
  <a class="gcard" href="docs.en.html">
    <h3>Docs &amp; Notes</h3>
    <p>The full plan, technical notes, 24 real pitfalls, and the key lessons. The most valuable part
    for anyone running an experiment like this.</p>
  </a>
  <a class="gcard" href="source.en.html">
    <h3>Source</h3>
    <p>Every experiment script, config file and data sample — click to read.</p>
  </a>
</div>

<h2>What's in the project</h2>
<div class="filetree">Local-Model-Lab/
│
├── base/                  ← original small student (Qwen3.5-2B, read-only, never modified)
├── branch-A-distill/      ← everything on the "copy the homework" branch (incl. trained A1)
├── branch-B-rl/           ← everything on the "practice solo" branch (incl. trained B / C)
│
├── shared/                ← shared by both branches
│   ├── results/           ★ evaluation reports and all raw data
│   └── bench_stratified.jsonl  ★ the frozen exam (52 problems)
│
├── scripts/               ← all programs (generate / screen / train / evaluate / report)
├── docs/                  ← notes (how it works, pitfalls, lessons learned)
├── configs/               ← every parameter setting
│
├── README.md              ← the spec for technical readers
├── 完整计划.md            ← the original full plan (Chinese)
├── 成果展示.md            ← the first round's results (Chinese)
└── 项目全貌.html / .md    ← the non-technical write-up (Chinese)</div>

<p style="font-size:13.5px;color:var(--muted);margin-top:16px">
  Items marked ★ are the most worth reading. <b>The model weight files (.safetensors, ~22 GB) are not
  included on this site</b> — they are neither practical to ship online nor necessary to understand
  the experiment.</p>

<h2>How it was done</h2>
<p>About 40 hours of machine time from start to conclusion: problem generation and screening (7 hours)
→ freeze the exam → train the three branches → one uniform exam (936 attempts) → auto-generated
report. There was one mishap along the way: on the first exam run a bug meant only the basic tier
made it into the paper, so the two hardest tiers were never tested at all. A second sitting produced
the complete results. That story is in the <a href="pitfalls.en.html">Pitfalls Log</a>.</p>
"""

# --------------------------------------------------------------------------- #
# 文档索引页
# --------------------------------------------------------------------------- #
DOCS_INDEX_ZH = """
<h1>文档笔记</h1>
<p class="lead">这个项目留下的全部文字记录。技术细节、设计取舍、以及踩过的坑 ——
做实验最值钱的部分其实在后两篇。</p>
<div class="grid">
%s
</div>
<div class="hl">
  <p><b>如果你是同学、想学怎么做这类实验</b>，建议顺序是：
  先看<a href="theory.html">原理笔记</a>搞懂蒸馏和强化学习的区别，
  再看<a href="lessons.html">经验教训总纲</a>了解哪些参数会坑人，
  最后翻<a href="pitfalls.html">踩坑记录</a> —— 那 24 个坑基本覆盖了在小机器上做这件事会遇到的全部麻烦。</p>
</div>
"""

DOCS_INDEX_EN = """
<h1>Docs &amp; Notes</h1>
<p class="lead">Everything written down over the course of this project: the technical details, the
design trade-offs, and the things that went wrong. The last two are the most valuable part for
anyone actually running an experiment.</p>
<div class="grid">
%s
</div>
<div class="hl">
  <p><b>If you're another student wanting to run something like this</b>, the suggested order is:
  start with the <a href="theory.en.html">technical notes</a> to get the difference between
  distillation and reinforcement learning, then read the
  <a href="lessons.en.html">key lessons</a> to learn which parameters will bite you, and finally go
  through the <a href="pitfalls.en.html">Pitfalls Log</a> — those 24 pitfalls cover essentially every
  problem you will hit doing this on a small machine.</p>
</div>
"""


def docs_card(href, title, desc):
    return '<a class="gcard" href="%s"><h3>%s</h3><p>%s</p></a>' % (href, esc(title), esc(desc))


# --------------------------------------------------------------------------- #
# 源码页
# --------------------------------------------------------------------------- #
def build_source_body(lang, py_files, sh_files, entry_files):
    groups = []
    for title, blurb, files in [
        (("训练与评测（Python）" if lang == "zh" else "Training &amp; evaluation (Python)"),
         ("出题、筛选、训练、评测、生成报告的全部 Python 程序。" if lang == "zh"
          else "Every Python program: problem generation, screening, training, evaluation, reporting."),
         py_files),
        (("流程控制（Shell）" if lang == "zh" else "Process control (Shell)"),
         ("环境检查、启动/暂停、进度查看、夜间长跑等 shell 脚本。" if lang == "zh"
          else "Shell scripts for environment checks, start/pause, progress, and overnight runs."),
         sh_files),
        (("入口与配置" if lang == "zh" else "Entry point &amp; config"),
         ("一句话向模型提问的入口脚本，以及全局配置（所有参数的事实来源）。" if lang == "zh"
          else "The one-line prompt entry script, plus the global config (single source of truth for all parameters)."),
         entry_files),
    ]:
        items = []
        for p in files:
            if not p.exists():
                continue
            code = p.read_text(encoding="utf-8", errors="replace")
            size = p.stat().st_size
            size_s = ("%.1f KB" % (size / 1024)) if size < 1048576 else ("%.1f MB" % (size / 1048576))
            desc = describe(p) if lang == "zh" else ""
            summary = '<span class="m2">%s%s</span>%s' % (
                size_s, (" · " + esc(desc)) if desc else "", esc(p.name))
            items.append('<details class="src"><summary>%s</summary><pre><code>%s</code></pre></details>'
                         % (summary, esc(code)))
        groups.append('<h2>%s</h2><p style="font-size:14px;color:var(--muted)">%s</p>%s'
                      % (title, blurb, "".join(items)))

    if lang == "zh":
        head = """
<h1>源码</h1>
<p class="lead">全部实验脚本、配置与数据样例，点开任意一项即可阅读。
共 %d 个 Python 程序、%d 个 Shell 脚本，以及一个全局配置文件。</p>
<p style="font-size:13.5px;color:var(--muted)">
  想看数据本身（考卷、原始答题记录）的话，它们都在项目的
  <code>shared/</code> 目录下：<code>bench_stratified.jsonl</code> 是冻结的 52 道考卷，
  <code>results/eval_raw.jsonl</code> 是 1014 条原始作答记录。</p>
""" % (len(py_files), len(sh_files))
    else:
        head = """
<h1>Source</h1>
<p class="lead">Every experiment script, config and data sample. Click any entry to read it.
%d Python programs, %d shell scripts, and one global config file.</p>
<p style="font-size:13.5px;color:var(--muted)">
  <b>Note:</b> the code is kept exactly as written in the project, so its inline comments are in
  Chinese. Everything else on this site has an English version.</p>
<p style="font-size:13.5px;color:var(--muted)">
  To see the data itself (the exam paper and the raw responses), look in the project's
  <code>shared/</code> directory: <code>bench_stratified.jsonl</code> is the frozen 52-problem exam,
  and <code>results/eval_raw.jsonl</code> holds all 1014 raw attempts.</p>
""" % (len(py_files), len(sh_files))
    return head + "".join(groups)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    written = []

    def emit(page_key, lang, content):
        name = PAGE_FILES[page_key][1 if lang == "en" else 0]
        (OUT / name).write_text(clean(content), encoding="utf-8")
        written.append(name)

    # ---------------- 首页 ----------------
    emit("index", "zh", build_page("index", "zh",
         "本地模型实验室 · 一个 AI 学生能不能超过它的老师", INDEX_ZH))
    emit("index", "en", build_page("index", "en",
         "Local Model Lab · Can a Small AI Student Surpass Its Teacher?", INDEX_EN))

    # ---------------- 已有自包含 HTML 页面：注入导航 ----------------
    for page_key, zh_path, en_path in [
        ("overview", ROOT / "项目全貌.html", I18N / "overview.en.html"),
        ("report", ROOT / "shared/results/横评报告.html", I18N / "report.en.html"),
    ]:
        if zh_path.exists():
            emit(page_key, "zh", inject_nav(zh_path.read_text(encoding="utf-8"),
                                            page_key, "zh"))
        en_raw = read_src(en_path)
        if en_raw:
            emit(page_key, "en", inject_nav(en_raw, page_key, "en"))

    # ---------------- Markdown 文档 ----------------
    for d in DOCS:
        zh_text = read_src(d["zh_src"])
        if zh_text:
            emit(d["key"], "zh", build_page(d["key"], "zh",
                 "%s · 本地模型实验室" % d["zh_title"],
                 '<div class="doc">%s</div>' % rewrite_links(md2html(zh_text), "zh")))
        en_text = read_src(d["en_src"])
        if en_text:
            emit(d["key"], "en", build_page(d["key"], "en",
                 "%s · Local Model Lab" % d["en_title"],
                 '<div class="doc">%s</div>' % rewrite_links(md2html(en_text), "en")))

    # ---------------- 文档索引页 ----------------
    zh_cards = "".join(docs_card(PAGE_FILES[d["key"]][0], d["zh_title"], d["zh_desc"]) for d in DOCS)
    en_cards = "".join(docs_card(PAGE_FILES[d["key"]][1], d["en_title"], d["en_desc"]) for d in DOCS)
    emit("docs", "zh", build_page("docs", "zh", "文档笔记 · 本地模型实验室",
                                  DOCS_INDEX_ZH % zh_cards))
    emit("docs", "en", build_page("docs", "en", "Docs & Notes · Local Model Lab",
                                  DOCS_INDEX_EN % en_cards))

    # ---------------- 源码页 ----------------
    py_files = sorted((ROOT / "scripts").glob("*.py"))
    sh_files = sorted((ROOT / "scripts").glob("*.sh"))
    entry_files = [ROOT / "ask.sh", ROOT / "configs/lab.config.json"]
    emit("source", "zh", build_page("source", "zh", "源码 · 本地模型实验室",
                                    build_source_body("zh", py_files, sh_files, entry_files)))
    emit("source", "en", build_page("source", "en", "Source · Local Model Lab",
                                    build_source_body("en", py_files, sh_files, entry_files)))

    # ---------------- 汇总 ----------------
    total = sum((OUT / f).stat().st_size for f in written)
    zh_n = sum(1 for f in written if not f.endswith(".en.html"))
    en_n = sum(1 for f in written if f.endswith(".en.html"))
    print("站点已生成：%s" % OUT)
    print("页面数：%d（中文 %d + 英文 %d），总体积：%.1f KB"
          % (len(written), zh_n, en_n, total / 1024))
    for f in sorted(written):
        print("   %-20s %8.1f KB" % (f, (OUT / f).stat().st_size / 1024))

    if MISSING:
        print("\n⚠️  以下英文源文件缺失，对应英文页面未生成：")
        for m in MISSING:
            print("     %s" % m)
    else:
        print("\n✅ 中英双语全部页面均已生成。")


if __name__ == "__main__":
    main()
