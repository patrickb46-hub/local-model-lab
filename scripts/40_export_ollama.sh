#!/usr/bin/env bash
# =============================================================================
# 40_export_ollama.sh —— 阶段 4：把训出来的 LoRA 合并进基座，导进 Ollama 日常用
#
# 路径选择（本机现实决定的）：
#   常规做法是 merge → 转 GGUF → ollama create。但本机：
#     · cmake 缺失，llama.cpp 编译不出来
#     · GitHub 直连只有 3.5 KB/s，克隆仓库等于不可用
#   所以优先走 **Ollama 原生 safetensors 导入**（Modelfile 里 FROM 一个权重目录，
#   由 Ollama 内置的转换器自己处理）——省掉整个 GGUF 工具链。
#   若 Ollama 不认这个架构，脚本会明确告诉你该怎么走 llama.cpp 那条备选路。
#
# 用法：
#   ./40_export_ollama.sh A          # 导出支线 A（蒸馏）
#   ./40_export_ollama.sh B          # 导出支线 B（RL）
#   ./40_export_ollama.sh both
# =============================================================================
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 解释器必须用 mlab-conda：旧的 envs/mlab 里是 torch 2.2.2 + 不支持 qwen3_5 的
# transformers，拿它加载 Qwen3.5 必崩（这是本机环境迁移后留下的坑）。
PY="${MLAB_PYTHON:-$HOME/.workbuddy/binaries/python/envs/mlab-conda/bin/python}"
WHICH="${1:-both}"

if [ ! -x "$PY" ]; then
    echo "✗ 找不到 mlab 解释器：$PY"
    exit 1
fi

export_args() {
    local tag="$1" branch="$2"
    local ADAPTER="$LAB_ROOT/$branch/adapter"
    local MERGED="$LAB_ROOT/$branch/merged"

    echo "──────────────────────────────────────────────────────────────"
    echo "  导出 $tag   支线目录 $branch"
    echo "──────────────────────────────────────────────────────────────"

    if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
        echo "  ✗ 没有训练产物（${ADAPTER}），跳过"
        return 1
    fi

    # ---------- 1. 合并 LoRA 到基座 ----------
    echo "  [1/3] 合并 LoRA → $MERGED"
    "$PY" - "$branch" "$ADAPTER" "$MERGED" "$LAB_ROOT" <<'PYEOF'
import json, sys
from pathlib import Path
branch, adapter, merged, lab_root = sys.argv[1:5]
sys.path.insert(0, str(Path(lab_root) / "scripts"))
from common import load_config
import labmodel as lm

cfg = load_config()
src = Path(lab_root) / branch / "model"
print(f"    基座 {src}")

# 【为什么用 bfloat16】合并只是 W + BA 的加法，精度需求低。
# 若用 fp32：需要同时持有「底座 8.8 GB + 合并后的副本 8.8 GB」≈ 17.6 GB，
# 超过本机 16 GB 物理内存 —— 必然被 OOM 杀掉。
# 注意这里只做前向合并、不涉及反向，所以 bf16 没有「反向慢 200 倍」的问题。
model, info = lm.load_model(str(src), dtype="bfloat16",
                           drop_vision=False,      # 导出时保留 vision，保持模型完整
                           trust_remote_code=False)
from peft import PeftModel
print(f"    挂载适配器 {adapter}")
model = PeftModel.from_pretrained(model, adapter)
import resource as _r
print(f"    当前峰值内存 {_r.getrusage(_r.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB"
      f"，开始合并（bf16 预计约 9 GB）…")
merged_model = model.merge_and_unload()
Path(merged).mkdir(parents=True, exist_ok=True)
# 分片保存：避免序列化单个超大文件时再吃一份内存
merged_model.save_pretrained(str(merged), safe_serialization=True,
                             max_shard_size="2GB")
tok = lm.load_tokenizer(str(src))
tok.save_pretrained(str(merged))
print(f"    ✓ 已写出 {merged}")
PYEOF

    # ---------- 2. 写 Modelfile ----------
    echo "  [2/3] 生成 Modelfile"
    cat > "$MERGED/Modelfile" <<EOF
FROM $MERGED

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "【答案】"

SYSTEM """你是一位耐心的初中数学老师。请解答用户给出的数学小题。
必须严格按以下固定格式作答，不要添加任何额外内容：
【思路】用一句话点明解题方向。
【步骤】
1. 第一步……
2. 第二步……
【答案】只写最终答案。
"""
EOF
    echo "    ✓ $MERGED/Modelfile"

    # ---------- 3. 导入 Ollama ----------
    echo "  [3/3] 导入 Ollama：$tag"
    if ! command -v ollama >/dev/null 2>&1; then
        echo "  ✗ 未找到 ollama 命令"
        return 1
    fi
    if ollama create "$tag" -f "$MERGED/Modelfile"; then
        echo "  ✓ 完成：ollama run $tag"
    else
        echo "  ✗ Ollama 未能识别该架构。备选路径："
        echo "     1) 装 cmake：brew install cmake"
        echo "     2) 取 llama.cpp 的 convert_hf_to_gguf.py（可走 gh-proxy）"
        echo "     3) python convert_hf_to_gguf.py $MERGED --outfile $branch/model-f16.gguf"
        echo "     4) Modelfile 改为 FROM ./model-f16.gguf 后重新 ollama create"
        return 1
    fi
}

RC=0
case "$WHICH" in
    A|a)   export_args "mylab-distill" "branch-A-distill" || RC=1 ;;
    B|b)   export_args "mylab-rl"      "branch-B-rl"      || RC=1 ;;
    both)  export_args "mylab-distill" "branch-A-distill" || RC=1
           export_args "mylab-rl"      "branch-B-rl"      || RC=1 ;;
    *)     echo "用法: $0 [A|B|both]"; exit 2 ;;
esac

echo
ollama list 2>/dev/null | head -12 || true
exit $RC
