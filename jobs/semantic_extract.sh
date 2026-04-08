#!/bin/bash
# SLURM array job: extract semantic annotations using local vLLM server
#
# Each node starts its own vLLM instance on localhost:8000, runs extraction
# against it, then shuts it down. The trap ensures cleanup on crash or timeout.
#
# Run AFTER scene_graphs.sh completes AND merge.py --stage scene_graphs has been run.
#
# Submit:
#   sbatch jobs/semantic_extract.sh
#
# After all tasks finish:
#   uv run python scripts/merge.py \
#     --stage semantic_annotations \
#     --data-dir /data/user_data/ethanchi/laion/semantic_10k

#SBATCH --job-name=semantic_extract
#SBATCH --array=0-7               # 8 shards
#SBATCH --gres=gpu:1              # vLLM needs a GPU
#SBATCH --time=12:00:00
#SBATCH --output=logs/semantic_%A_%a.log
#SBATCH --error=logs/semantic_%A_%a.log

NUM_SHARDS=8    # keep in sync with --array upper bound + 1

# ── Start vLLM server in background ──────────────────────────────────────────
uvx vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8000 \
    --max-model-len 32768 \
    --trust-remote-code &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# ── Wait for vLLM to be ready ─────────────────────────────────────────────────
echo "Waiting for vLLM server (PID $VLLM_PID)..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready."

# ── Run extraction ────────────────────────────────────────────────────────────
uv run python scripts/extract_semantic.py \
    --config configs/semantic_10k.toml \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS" \
    --concurrency 8    # concurrent requests to local vLLM; tune based on throughput
