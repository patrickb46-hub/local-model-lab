#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common.py —— 本地模型实验室公共库

两条支线（蒸馏 A / 强化学习 B）共用这套地基：
  * 配置加载
  * 固定输出格式的解析与判分（【思路】/【步骤】/【答案】）
  * 答案归一化与等价比较
  * Ollama teacher 调用（带 think=false、重试、断点续跑）
  * jsonl 读写

设计原则：不依赖除标准库以外的任何东西（torch/transformers 只在实际训练脚本里用），
这样 09/10/30 这类"纯逻辑"脚本可以在任何 Python 上跑。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# 路径与配置
# --------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
LAB_ROOT = SCRIPTS_DIR.parent
CONFIG_PATH = LAB_ROOT / "configs" / "lab.config.json"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def lab_path(cfg: Dict[str, Any], key: str) -> Path:
    """按 config 里 paths 的短名解析成绝对路径。"""
    rel = cfg["paths"][key]
    p = Path(rel)
    return p if p.is_absolute() else (LAB_ROOT / rel)


def resolve_python(cfg: Dict[str, Any], prefer: Optional[str] = None) -> str:
    """
    挑一个可用的解释器。默认用 config.runtime.python（conda 环境，带现代 torch），
    不存在时退到 python_fallback。不存在的路径直接跳过，避免脚本在半路才炸。
    """
    import shutil
    rt = cfg.get("runtime", {})
    cands = []
    if prefer:
        cands.append(prefer)
    cands += [rt.get("python"), rt.get("python_fallback"), rt.get("python_system")]
    for c in cands:
        if not c:
            continue
        # 允许 config 里写 ~ 开头的路径（便于跨机器移植），这里统一展开
        c = os.path.expanduser(os.path.expandvars(str(c)))
        if Path(c).exists():
            return c
    return sys.executable


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_runtime_env(cfg: Dict[str, Any]) -> None:
    """把 config.runtime.env 注入 os.environ，并限制 torch 线程数。"""
    for k, v in (cfg.get("runtime", {}).get("env") or {}).items():
        os.environ.setdefault(k, str(v))
    n = cfg.get("runtime", {}).get("torch_threads")
    if n:
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))


# --------------------------------------------------------------------------
# jsonl 读写
# --------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# 固定输出格式：解析
# --------------------------------------------------------------------------

SEC_IDEA = "【思路】"
SEC_STEPS = "【步骤】"
SEC_ANSWER = "【答案】"
SECTION_ORDER = [SEC_IDEA, SEC_STEPS, SEC_ANSWER]

_FULLWIDTH_MAP = {
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "＋": "+", "－": "-", "＝": "=", "×": "*", "÷": "/",
    "（": "(", "）": ")", "，": ",", "：": ":", "；": ";",
    "．": ".", "／": "/", "　": " ",
}


def to_halfwidth(s: str) -> str:
    out = []
    for ch in s:
        if ch in _FULLWIDTH_MAP:
            out.append(_FULLWIDTH_MAP[ch])
        else:
            # 全角数字/字母的统一处理
            if unicodedata.east_asian_width(ch) == "F" and ch.isalnum():
                out.append(unicodedata.normalize("NFKC", ch))
            else:
                out.append(ch)
    return "".join(out)


def _strip_math_decoration(s: str) -> str:
    """去掉 LaTeX / markdown 包裹，只留下内容。"""
    s = s.strip()
    # \boxed{...}  \text{...}
    for _ in range(4):
        s2 = re.sub(r"\\(?:boxed|text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", s)
        if s2 == s:
            break
        s = s2
    s = s.replace("$$", "").replace("$", "")
    s = s.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    s = s.replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # \frac{a}{b} -> a/b
    for _ in range(4):
        s2 = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
        if s2 == s:
            break
        s = s2
    s = s.replace("\\", "")
    # markdown 强调
    s = s.replace("**", "").replace("`", "")
    return s.strip()


_TRAILING_NOISE = re.compile(
    r"[\s。．\.，,；;！!？?、]+$"
)
_UNIT_SUFFIX = re.compile(
    r"(平方厘米|平方分米|平方米|平方千米|立方厘米|立方分米|立方米|"
    r"厘米|分米|千米|毫米|米|克|千克|吨|分钟|小时|秒|天|元|角|分|个|只|本|人|"
    r"度|岁|倍|次|棵|件|页|字|cm2|cm²|m2|m²|cm|dm|km|mm|kg|g|s|h|min)$"
)


def normalize_answer(s: str) -> str:
    """把答案归一化成可比较的规范串。"""
    if s is None:
        return ""
    s = to_halfwidth(str(s))
    s = _strip_math_decoration(s)
    # 只取第一行（答案区可能带解释）
    s = s.split("\n")[0]
    # 去前缀词
    s = re.sub(r"^(答案|答|所以|因此|故|结果为|等于)\s*[:：=]?\s*", "", s)
    s = s.strip().strip("。．.，,；;!！?？")
    s = re.sub(r"\s+", "", s)
    if len(s) > 80:
        s = s[:80]
    return s


def parse_number(s: str) -> Optional[Fraction]:
    """尝试把字符串解析成分数/数值。支持 a/b、-a/b、小数、百分数、带单位。"""
    if not s:
        return None
    t = s.strip()
    t = re.sub(r"^[xyabn=:：]+\s*", "", t)
    # 百分数
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", t)
    if m:
        return Fraction(m.group(1)) / 100
    # 去掉单位
    while True:
        m = _UNIT_SUFFIX.search(t)
        if not m:
            break
        t = t[: m.start()].strip()
    t = t.strip()
    # 纯数值
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return Fraction(t)
    # 分数
    if re.fullmatch(r"-?\d+\s*/\s*\d+", t):
        a, b = re.split(r"/", t)
        try:
            b_i = int(b)
            if b_i == 0:
                return None
            return Fraction(int(a), b_i)
        except Exception:
            return None
    # 带分数 1又1/2 或 1 1/2
    m = re.fullmatch(r"(-?\d+)[又\s]+(\d+)/(\d+)", t)
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if den == 0:
            return None
        sign = -1 if whole < 0 else 1
        return sign * (abs(whole) + Fraction(num, den))
    return None


def answer_equal(pred: str, gold: str, tol: float = 1e-6) -> bool:
    """判断预测答案与标准答案是否等价。"""
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if not np_:
        return False
    if np_ == ng:
        return True
    fp, fg = parse_number(np_), parse_number(ng)
    if fp is not None and fg is not None:
        try:
            return abs(float(fp) - float(fg)) <= tol
        except Exception:
            return False
    # 数值 + 容差（浮点小数）
    m1 = re.fullmatch(r"-?\d+\.\d+", np_)
    m2 = re.fullmatch(r"-?\d+\.\d+", ng)
    if m1 and m2:
        return abs(float(np_) - float(ng)) <= 1e-4
    # 单位不同但数值相同的情况已在 parse_number 处理；剩下做宽松包含
    if len(np_) >= 2 and (np_ in ng or ng in np_):
        return True
    return False


# --------------------------------------------------------------------------
# 固定输出格式：判分
# --------------------------------------------------------------------------

def split_think(text: str) -> Tuple[str, str]:
    """
    把 `<think>…</think>` 推理块与正式回答分开，返回 (正式回答, 推理内容)。

    为什么必须有这一步：开启 thinking 后，模型在推理块里**也会写出
    【思路】【步骤】【答案】** 这些标记（它在自言自语地套格式）。
    不先剥掉，判分器会把推理过程当成正式答案，指标全废。
    """
    if not text:
        return "", ""
    t = text
    i = t.rfind("</think>")
    if i >= 0:
        return t[i + len("</think>"):].strip(), t[:i].strip()
    j = t.find("<think>")
    if j >= 0:
        # 未闭合：多半是 max_new_tokens 截断，推理没结束 → 没有正式回答
        return "", t[j + len("<think>"):].strip()
    return t.strip(), ""


_ANS_STRONG = re.compile(r"(?:最终答案|参考答案|答案)\s*[:：是]?\s*([^\n。；;]{1,60})")
_ANS_WEAK = re.compile(r"(?:结果为|结果是|所以|因此|故|答)\s*[:：是]?\s*([^\n。；;]{1,60})")


def _loose_answer_extract(text: str) -> str:
    """
    兜底答案提取：模型没有使用【答案】标记时，尽量把答案摘出来。

    为什么必须要有这一步：
      官方原版模型（qwen3.5:2b / 4b / 9b）没有经过本项目的格式微调，
      很可能写成「答案是 2」「**答案：** x = 2」甚至只有「x = 2」。
      如果因为「没出现【答案】二字」就把它们判为全错，
      那「正确率」就退化成「格式遵从率」的复制品，
      横评会得出「官方 4B 远不如微调后的 2B」这种**荒谬且方向相反**的结论。

    边界（很重要）：
      本函数只影响「答案提取」，「格式遵从率」仍由 parse_response 严格判定
      （必须三段齐全且顺序正确）。这样两个指标才是彼此独立的。

    策略按优先级：
      1. 强线索（答案 / 最终答案）之后、且能解析成数值的片段
      2. 等式右端（`= 2`）且能解析成数值
      3. 弱线索（所以 / 因此 / 结果为 …）之后且能解析成数值
      4. 强线索 / 弱线索的原文片段（答案是表达式时）
      5. 最后一个等式右端
      6. 退回最后一行 —— 但只在它短得像答案时（≤20 字符），否则宁可放弃，
         免得把一整段废话当成答案造成误判
    """
    if not text:
        return ""
    t = text.strip()
    strong = list(_ANS_STRONG.finditer(t))
    weak = list(_ANS_WEAK.finditer(t))
    eqs = re.findall(r"=\s*([^\n=]{1,40})", t)

    def _numeric(c: str) -> bool:
        return bool(c) and parse_number(normalize_answer(c)) is not None

    for m in reversed(strong):                      # 1
        c = m.group(1).strip()
        if _numeric(c):
            return c
    for c in reversed(eqs):                         # 2
        c = c.strip()
        if _numeric(c):
            return c
    for m in reversed(weak):                        # 3
        c = m.group(1).strip()
        if _numeric(c):
            return c
    if strong:                                      # 4
        return strong[-1].group(1).strip()
    if weak:                                        # 5
        return weak[-1].group(1).strip()
    if eqs:                                         # 6
        return eqs[-1].strip()
    last = [x for x in t.split("\n") if x.strip()]  # 7
    if last:
        cand = last[-1].strip()
        # 必须短、且含数字 —— 否则「抱歉，我不会做」这种话也会被当成答案
        if len(cand) <= 20 and re.search(r"\d", cand):
            return cand
    return ""


def parse_response(text: str) -> Dict[str, Any]:
    """
    解析模型输出，返回:
      {
        "format_ok": bool,       # 三个 section 齐全且顺序正确
        "has_idea": bool, "has_steps": bool, "has_answer": bool,
        "steps": [str, ...],     # 拆出来的编号步骤
        "answer_raw": str,       # 【答案】之后的内容（原样，第一行）
        "answer_norm": str,      # 归一化后的答案
        "extra_tail": str,       # 【答案】之后多余的废话
        "thinking": str,         # 推理块内容（如果有）
        "thinking_chars": int,   # 推理长度，用于比较 think 开/关
      }

    注意：永远先剥掉 `<think>…</think>` 再判分（见 split_think 的说明）。
    """
    raw, think = split_think(text or "")
    res: Dict[str, Any] = {
        "format_ok": False,
        "has_idea": False,
        "has_steps": False,
        "has_answer": False,
        "steps": [],
        "answer_raw": "",
        "answer_norm": "",
        "answer_source": "none",   # tag=用了【答案】标记 / fallback=兜底摘取 / none=没拿到
        "extra_tail": "",
        "thinking": think,
        "thinking_chars": len(think),
    }
    if not raw:
        return res

    idx = {}
    for sec in SECTION_ORDER:
        i = raw.find(sec)
        if i >= 0:
            idx[sec] = i
    res["has_idea"] = SEC_IDEA in idx
    res["has_steps"] = SEC_STEPS in idx
    res["has_answer"] = SEC_ANSWER in idx

    if res["has_answer"]:
        tail = raw[idx[SEC_ANSWER] + len(SEC_ANSWER):].strip()
        first = tail.split("\n")[0].strip()
        res["answer_raw"] = first
        res["answer_norm"] = normalize_answer(first)
        res["answer_source"] = "tag"
    else:
        # 兜底：模型没用我们的标记（官方原版多半如此）。
        # 关键：只补「答案提取」，不动 format_ok —— 格式遵从率仍严格要求三段齐全。
        # 这样「答得对不对」与「格不格式」才是两个独立维度，横评才公平。
        guess = _loose_answer_extract(raw)
        res["answer_raw"] = guess
        res["answer_norm"] = normalize_answer(guess)
        res["answer_source"] = "fallback" if guess else "none"

    if res["has_steps"]:
        start = idx[SEC_STEPS] + len(SEC_STEPS)
        end = idx.get(SEC_ANSWER, len(raw))
        block = raw[start:end]
        steps = re.findall(r"^\s*(?:\d+[\.、\)]|[一二三四五六七八九十]+[、\.\)])\s*(.+?)\s*$",
                           block, flags=re.MULTILINE)
        res["steps"] = [s.strip() for s in steps if s.strip()]

    # 顺序正确性
    if res["has_idea"] and res["has_steps"] and res["has_answer"]:
        ordered = idx[SEC_IDEA] < idx[SEC_STEPS] < idx[SEC_ANSWER]
        res["format_ok"] = bool(ordered)

    return res


def format_compliance(text: str) -> bool:
    return parse_response(text)["format_ok"]


# --------------------------------------------------------------------------
# 奖励函数（RL 支线专用，全可验证、不需要奖励模型）
# --------------------------------------------------------------------------

def compute_reward(text: str, gold: str, weights: Dict[str, float],
                   max_answer_chars: int = 64) -> Tuple[float, Dict[str, float]]:
    """返回 (总奖励, 分项明细)。"""
    p = parse_response(text)
    parts = {
        "correct": 0.0,
        "format": 0.0,
        "steps": 0.0,
        "overlong_penalty": 0.0,
    }
    if p["format_ok"]:
        parts["format"] = 1.0
    if p["has_steps"] and len(p["steps"]) >= 1:
        parts["steps"] = 1.0
    if p["answer_norm"] and answer_equal(p["answer_norm"], gold):
        parts["correct"] = 1.0
    if p["has_answer"] and len(p["answer_raw"]) > max_answer_chars:
        parts["overlong_penalty"] = 1.0

    total = 0.0
    for k, w in weights.items():
        total += w * parts.get(k, 0.0)
    return total, parts


# --------------------------------------------------------------------------
# Ollama teacher
# --------------------------------------------------------------------------

def ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]],
                options: Optional[Dict[str, Any]] = None,
                think: bool = False,
                timeout: int = 600,
                retry: int = 3) -> Optional[str]:
    """
    调用 Ollama /api/chat。stream=False。
    关键：Qwen3.5 默认开思考，必须显式传 think=false，否则 content 为空、内容全在 thinking 通道。
    """
    import urllib.request
    import urllib.error

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": bool(think),
        "options": options or {},
    }
    data = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/api/chat"

    last_err = None
    for attempt in range(1, retry + 1):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            msg = body.get("message") or {}
            content = msg.get("content") or ""
            if not content.strip() and msg.get("thinking"):
                # 兜底：万一还是走了思考通道
                content = msg.get("thinking") or ""
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 * attempt, 8))
    print(f"[ollama_chat] 失败({model}): {last_err}", file=sys.stderr)
    return None


def ollama_health(base_url: str) -> Optional[List[str]]:
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=10) as r:
            d = json.loads(r.read().decode("utf-8"))
        return [m["name"] for m in d.get("models", [])]
    except Exception:
        return None


def cloud_chat(base_url: str, model: str, messages: List[Dict[str, str]],
               api_key: str, max_tokens: int = 1024, temperature: float = 0.0,
               think: bool = False, timeout: int = 300, retry: int = 3,
               ) -> Tuple[Optional[str], Optional[int], int]:
    """
    调用 Anthropic 兼容的 Messages API（本项目接的是 DeepSeek 的 /anthropic 端点）。

    返回 (正文, 输出 token 数, 推理字符数)；失败返回 (None, None, 0)。

    两个必须知道的点（都是实测出来的，不是照抄文档）：
      1. 这是**推理模型**，思考内容放在**独立的内容块**里
         （content 里 type == "thinking" 的块），不是 <think> 标签。
         所以正文必须按块类型挑，不能直接拿 content[0] —— 否则会把
         英文思维链当成正式回答，判分全废。
      2. think=False 时必须显式传 thinking={"type":"disabled"}。
         不传的话思考会吃掉全部 max_tokens，正文一个字都出不来
         （第一次实测就是 max_tokens=120 全被 thinking 吃光）。
    """
    import urllib.error
    import urllib.request

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": messages,
    }
    if not think:
        payload["thinking"] = {"type": "disabled"}

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    last_err: Any = None
    for attempt in range(1, retry + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            texts, thinks = [], []
            for blk in (body.get("content") or []):
                t = blk.get("type")
                if t == "text":
                    texts.append(blk.get("text") or "")
                elif t in ("thinking", "redacted_thinking"):
                    thinks.append(blk.get("thinking") or "")
            ntok = (body.get("usage") or {}).get("output_tokens")
            return "".join(texts).strip(), ntok, len("".join(thinks))
        except urllib.error.HTTPError as e:  # 4xx/5xx 要把响应体带出来
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:200]
            except Exception:  # noqa: BLE001
                pass
            last_err = f"HTTP {e.code} {detail}"
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(min(2 * attempt, 10))

    print(f"[cloud_chat] 失败({model}): {last_err}", file=sys.stderr)
    return None, None, 0


def cloud_health(base_url: str, model: str, api_key: str,
                 timeout: int = 60) -> Optional[str]:
    """
    探活：发一个最小请求。成功返回 None，失败返回错误说明字符串。
    目的是**快速失败** —— 别等横评跑到一半才发现 key 过期或端点写错。
    """
    if not api_key:
        return "API key 为空（检查环境变量是否注入）"
    txt, _, _ = cloud_chat(base_url, model, [{"role": "user", "content": "ping"}],
                           api_key=api_key, max_tokens=24,
                           think=False, timeout=timeout, retry=1)
    return None if txt is not None else "调用失败（见上方 cloud_chat 的 stderr）"


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------

def human_time(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m}m{s}s"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


def banner(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n  {title}\n{line}", flush=True)
