#!/usr/bin/env bash
# =============================================================================
# lint_shell.sh —— 全项目 shell 脚本体检
#
# 查三类问题：
#   1. 语法错误（bash -n）
#   2. ${VAR} 边界隐患 —— 写成 "$VAR（" 时，bash 会把紧随其后的**全角标点**
#      当成变量名的一部分，在 set -u 下直接 `unbound variable` 终止脚本。
#      本项目已因此踩坑 2 次（resume.sh、watchdog.sh），正确写法一律用 "${VAR}（"。
#      —— 这个坑靠记忆必重犯，所以做成工具每次跑一遍。
#   3. 在 set -u 脚本里用了未定义的常见变量名（粗查）
#
# 用法：bash scripts/lint_shell.sh
# =============================================================================
set -uo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB_ROOT"

FAIL=0
echo "════ shell 脚本体检 ════"

for f in scripts/*.sh; do
    name="$(basename "$f")"

    # --- 1. 语法 ---
    if ! bash -n "$f" 2>/dev/null; then
        echo "  ✗ 语法错误：$name"
        bash -n "$f" 2>&1 | sed 's/^/      /'
        FAIL=1
        continue
    fi

    # --- 2. $VAR 紧跟全角标点（unbound 隐患）---
    # 排除纯注释行，否则本脚本自己说明这个问题时也会被误判
    hits="$(grep -nE '\$[A-Za-z_][A-Za-z0-9_]*[（）：，。、「」；！？·…]' "$f" 2>/dev/null \
            | grep -vE '^[0-9]+:[[:space:]]*#' || true)"
    if [ -n "$hits" ]; then
        echo "  ✗ unbound 隐患：$name"
        echo "$hits" | sed 's/^/      /'
        echo "      → 改成 \${VAR} 形式"
        FAIL=1
        continue
    fi

    echo "  ok  $name"
done

echo "────────────────────────────────"
if [ "$FAIL" -eq 0 ]; then
    echo "全部通过。"
else
    echo "发现问题，请修复后重跑。"
fi
exit $FAIL
