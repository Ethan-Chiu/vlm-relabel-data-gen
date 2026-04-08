#!/bin/bash
#SBATCH --job-name=semantic
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2:00:00
#SBATCH --output=/home/ethanchi/data-gen-log/semantic/slurm-%j.log
#SBATCH --error=/home/ethanchi/data-gen-log/semantic/slurm-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ethanchi@andrew.cmu.edu

# --- 1. Environment Setup ---

echo "Running semantic extraction..."

export NUM_SHARDS=3
export SLURM_ARRAY_TASK_ID=0

# ── Start vLLM server in background ──────────────────────────────────────────
uvx vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8000 \
    --max-model-len 8192 \
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
    --concurrency 16  2>&1  # concurrent requests to local vLLM; tune based on throughput


echo "Job finished."
