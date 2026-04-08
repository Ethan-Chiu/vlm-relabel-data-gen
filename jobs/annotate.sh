#!/bin/bash
# SLURM array job: semantic caption generation, targeting 10k output rows
#
# Run AFTER semantic_extract.sh completes AND merge.py --stage semantic_annotations has been run.
#
# --limit 10000 is applied to the full joined dataset before shard splitting,
# so each shard processes ~(10000 / NUM_SHARDS) rows. Total output = 10k.
#
# Submit:
#   sbatch jobs/annotate.sh
#
# After all tasks finish:
#   uv run python scripts/merge.py \
#     --stage annotated \
#     --data-dir /data/user_data/ethanchi/laion/semantic_10k

#SBATCH --job-name=annotate_semantic
#SBATCH --array=0-7               # 8 shards
#SBATCH --time=4:00:00
#SBATCH --output=logs/annotate_%A_%a.log

NUM_SHARDS=8    # keep in sync with --array upper bound + 1

uv run python scripts/annotate.py \
  --config configs/semantic_10k.toml \
  --pipeline semantic \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  --concurrency 8 \
  --limit 10000    # 10k total rows split across all shards (~1250/shard at 8 shards)
