export NUM_SHARDS=4
export SLURM_ARRAY_TASK_ID=3

uv run python scripts/download.py \
    --config configs/semantic_10k.toml \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS"