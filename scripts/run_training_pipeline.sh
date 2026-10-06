#!/usr/bin/env bash
# 等 Qwen 难例重对齐跑完 → 合并出训练数据 → 启动 QLoRA 训练
#
#   bash scripts/run_training_pipeline.sh              # 全流程
#   bash scripts/run_training_pipeline.sh --set training.max_steps=50   # 冒烟
#
# 每一步都会写日志；训练可随时中断并用 --resume auto 续跑。
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./py
LOG=logs/pipeline.log
mkdir -p logs

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# 1) 等待重对齐进程结束（可中断，缓存会保留）
if pgrep -f "align_chunks_qwen.py" >/dev/null 2>&1; then
  log "等待 Qwen 重对齐结束…"
  while pgrep -f "align_chunks_qwen.py" >/dev/null 2>&1; do sleep 30; done
  log "重对齐进程已结束"
else
  log "没有正在运行的重对齐进程，直接进入合并"
fi

# 2) 合并缓存 → 修正后的 chunks + 训练格式
log "合并重对齐结果并生成训练数据…"
$PY scripts/align_chunks_qwen.py --merge-only --chat 2>&1 | tee -a "$LOG"
if [ ! -f data/processed/train.chunks.qwen.chat.jsonl ]; then
  log "错误：未生成训练数据，中止"
  exit 1
fi

# 3) 预检：加载 7B + LoRA + 数据集，但不训练（约 2 分钟，避免长跑后才发现问题）
log "预检：加载模型与数据（--dry-run）…"
if ! $PY scripts/train.py --config configs/train.scp.yaml --dry-run 2>&1 | tee -a "$LOG"; then
  log "错误：预检失败，已中止（未开始训练）"
  exit 2
fi
log "预检通过"

# 4) 训练
log "开始 QLoRA 训练…"
$PY scripts/train.py --config configs/train.scp.yaml "$@" 2>&1 | tee -a "$LOG"
log "完成。adapter 位于 outputs/qwen2.5-7b-scp-chunks/final_adapter"
