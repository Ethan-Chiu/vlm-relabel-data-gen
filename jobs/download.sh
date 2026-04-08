export SLURM_ARRAY_TASK_ID=0

NUM_SHARDS=4 uv run python scripts/download.py \
      --config configs/semantic_10k.toml \
      --shard-id \$SLURM_ARRAY_TASK_ID \
      --num-shards \$NUM_SHARDS"