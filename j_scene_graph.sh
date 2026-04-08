#!/bin/bash
#SBATCH --job-name=scenegraphextract
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2:00:00
#SBATCH --output=/home/ethanchi/data-gen-log/scene_graphs/slurm-%j.log
#SBATCH --error=/home/ethanchi/data-gen-log/scene_graphs/slurm-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ethanchi@andrew.cmu.edu

# --- 1. Environment Setup ---

echo "Running scene graph extraction..."

export NUM_SHARDS=3
export SLURM_ARRAY_TASK_ID=2

uv run python scripts/extract_scene_graphs.py \
  --config configs/semantic_10k.toml \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  --concurrency 1 \
  --limit 15000 2>&1


echo "Job finished."
