#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# progress_en.sh —— 英文版进度查看器（progress.sh --en 的薄包装）
#
# 为什么单独放一个文件：英文界面下不想每次多敲一个 --en。
# 所有逻辑都在 progress.sh 里，这里只是转发，避免两份代码各自漂移。
#
# 用法：
#   bash scripts/progress_en.sh        # 看一次
#   bash scripts/progress_en.sh -w     # 每 20 秒自动刷新
#   bash scripts/progress_en.sh -w 5   # 自定义刷新间隔
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DIR/progress.sh" --en "$@"
