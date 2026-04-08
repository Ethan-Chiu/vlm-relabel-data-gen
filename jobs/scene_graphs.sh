#!/bin/bash
# SLURM array job: extract scene graphs (GPU stage)
#
# Adjust --array and NUM_SHARDS to match the number of GPU nodes available.
# Both must stay in sync: --array=0-(N-1), NUM_SHARDS=N.
#
# Submit:
#   sbatch jobs/scene_graphs.sh
#
# After all tasks finish:
#   uv run python scripts/merge.py \
#     --stage scene_graphs \
#     --data-dir /data/user_data/ethanchi/laion/semantic_10k

#SBATCH --job-name=scene_graphs
#SBATCH --array=0-3               # 4 GPU nodes → shards 0,1,2,3
#SBATCH --gres=gpu:1              # 1 GPU per node (SAM vit_h needs ~7 GB VRAM)
#SBATCH --time=12:00:00
#SBATCH --output=logs/scene_%A_%a.log

NUM_SHARDS=4    # keep in sync with --array upper bound + 1

uv run python scripts/extract_scene_graphs.py \
  --config configs/semantic_10k.toml \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  --concurrency 1    # 1 worker per GPU (RAM++/GroundingDINO/SAM loaded once per worker)
