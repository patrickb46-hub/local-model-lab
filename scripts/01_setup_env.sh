#!/usr/bin/env bash
# =============================================================================
# 01_setup_env.sh —— 一次性搭建 mlab 训练环境（可重复执行，幂等）
#
# 为什么这么麻烦？因为本机是 2018 款 Intel Mac（x86_64），而：
#   1. PyTorch 官方在 2.2.2 之后不再发布 macOS x86_64 轮子 —— torch 天花板 = 2.2.2
#   2. torch 2.2.2 最高只支持 Python 3.12 —— 本机 managed Python 是 3.13，装不上
#   3. torch 2.2.2 是按 numpy 1.x 编译的 —— 必须把 numpy 钉在 <2，否则
#      "Numpy is not available"，numpy 互操作全废
#   4. 本机 PyPI 直连超时、清华源 403 —— 只有阿里云镜像通
#   5. HF 官方直连不通 —— 必须走 hf-mirror
#
# 于是：从 npmmirror 拉一个独立 CPython 3.12 装到 managed 目录，用它建 venv。
# =============================================================================
set -euo pipefail

PY_VER="3.12.15"
PBS_DATE="${PBS_DATE:-20261003}"          # python-build-standalone 的构建日期
PBS_BASE="https://registry.npmmirror.com/-/binary/python-build-standalone"

HOME_DIR="$HOME"
BIN_DIR="$HOME_DIR/.workbuddy/binaries/python"
VERS_DIR="$BIN_DIR/versions"
ENVS_DIR="$BIN_DIR/envs"
TARGET="$VERS_DIR/$PY_VER"
VENV="$ENVS_DIR/mlab"
PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/"

echo "=================================================================="
echo "  01_setup_env.sh  ·  mlab 训练环境"
echo "=================================================================="
echo "  目标 Python : $PY_VER"
echo "  目标 venv   : $VENV"
echo "  PyPI 镜像   : $PIP_INDEX"
echo

mkdir -p "$VERS_DIR" "$ENVS_DIR"

# ---------------------------------------------------------------------------
# 1. 独立 CPython 3.12
# ---------------------------------------------------------------------------
if [ -x "$TARGET/bin/python3" ] && "$TARGET/bin/python3" -c "import sys" 2>/dev/null; then
    echo "[1/4] Python $PY_VER 已存在，跳过下载"
else
    echo "[1/4] 下载独立 CPython ${PY_VER}（来自 npmmirror，避免 GitHub 龟速）"
    NEED_MIRROR=1
    if command -v brew >/dev/null 2>&1; then NEED_MIRROR=1; fi

    # 自动找最新的可用构建日期（若默认日期已失效）
    pick_url() {
        local d="$1"
        local enc="cpython-${PY_VER}%2B${d}-x86_64-apple-darwin-install_only.tar.gz"
        echo "$PBS_BASE/$d/$enc"
    }
    URL="$(pick_url "$PBS_DATE")"
    if ! curl -sSfI --max-time 20 "$URL" >/dev/null 2>&1; then
        echo "      默认日期 $PBS_DATE 失效，查询最新构建…"
        DATE_LIST="$(curl -sS --max-time 25 "$PBS_BASE/" \
            | python3 -c "import json,sys;d=json.load(sys.stdin);\
print('\n'.join(sorted([x['name'].rstrip('/') for x in d if x['type']=='dir'])[-8:]))")"
        for d in $(echo "$DATE_LIST" | tac); do
            if curl -sSfI --max-time 20 "$(pick_url "$d")" >/dev/null 2>&1; then
                URL="$(pick_url "$d")"; echo "      命中 $d"; break
            fi
        done
    fi

    TMP="$(mktemp -d)"
    echo "      $URL"
    curl -sSL --max-time 900 -o "$TMP/py.tar.gz" "$URL"
    tar -xzf "$TMP/py.tar.gz" -C "$TMP"
    rm -rf "$TARGET"
    mv "$TMP/python" "$TARGET"
    rm -rf "$TMP"
fi
"$TARGET/bin/python3" --version

# ---------------------------------------------------------------------------
# 2. venv
# ---------------------------------------------------------------------------
if [ -x "$VENV/bin/python" ]; then
    echo "[2/4] venv 已存在：$VENV"
else
    echo "[2/4] 创建 venv：$VENV"
    "$TARGET/bin/python3" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip -i "$PIP_INDEX"
"$VENV/bin/python" -m pip --version

# ---------------------------------------------------------------------------
# 3. 依赖栈（版本全部钉死，理由见文件头）
# ---------------------------------------------------------------------------
echo "[3/4] 安装依赖栈（torch 150MB+，需要几分钟）"
"$VENV/bin/python" -m pip install -i "$PIP_INDEX" \
    "torch==2.2.2" \
    "transformers==5.18.0" \
    "peft==0.21.2" \
    "trl==1.14.1" \
    "datasets==5.0.1" \
    "accelerate==1.15.0" \
    "safetensors==0.8.0" \
    "sentencepiece==0.2.2" \
    "numpy<2" \
    "huggingface_hub==1.33.0"

# ---------------------------------------------------------------------------
# 4. 验证
# ---------------------------------------------------------------------------
echo "[4/4] 验证"
"$VENV/bin/python" - <<'PY'
import importlib, sys
print("  python      ", sys.version.split()[0])
for m in ["torch", "transformers", "peft", "trl", "datasets", "accelerate",
          "safetensors", "sentencepiece", "numpy", "huggingface_hub"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m:16s} 导入失败: {e}")
        sys.exit(1)
import torch, numpy
print("  torch/numpy 互操作", torch.tensor(numpy.array([1.0])).item())
print("  MPS 可用        ", torch.backends.mps.is_available(), "(Intel 预期 False)")
PY

echo
echo "完成。训练用解释器："
echo "  $VENV/bin/python"
