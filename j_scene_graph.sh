#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/ethanchi/data-gen-log/scene_graphs/slurm-%j.log
#SBATCH --error=/home/ethanchi/data-gen-log/scene_graphs/slurm-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ethanchi@andrew.cmu.edu

# --- 1. Environment Setup ---

echo "Running scene graph extraction..."

export NUM_SHARDS=4
export SLURM_ARRAY_TASK_ID=0

uv run python scripts/extract_scene_graphs.py \
  --config configs/semantic_10k.toml \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  --concurrency 1  2>&1


echo "Job finished."
