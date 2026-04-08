#!/bin/bash
# SLURM array job: extract semantic annotations (API stage, no GPU needed)
#
# Run AFTER scene_graphs.sh completes AND merge.py --stage scene_graphs has been run.
#
# Adjust --array and NUM_SHARDS to the number of nodes you want to use.
# Tune --concurrency based on your API rate limits (8 is conservative).
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
#SBATCH --time=8:00:00
#SBATCH --output=logs/semantic_%A_%a.log

NUM_SHARDS=8    # keep in sync with --array upper bound + 1

uv run python scripts/extract_semantic.py \
  --config configs/semantic_10k.toml \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  --concurrency 8    # concurrent API calls per node; reduce if hitting rate limits
