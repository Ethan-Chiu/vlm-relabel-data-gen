# Distributed Workflow

This document covers running the pipeline across multiple machines that share a filesystem (NFS or similar).

## Design overview

- Each machine is assigned a **shard** (`--shard-id N --num-shards M`). Work is partitioned deterministically by `md5(filename) % num_shards`, so rows are always assigned to the same shard regardless of how many machines are used.
- Each machine writes its results to `data/_staging/` rather than the canonical parquet. After all shards finish, `merge.py` combines staging files into the canonical parquet and deletes them.
- Skip logic always reads the **canonical** parquet, not staging files. This means machines can fail, `num_shards` can change between runs, and limits can be extended — the next run always picks up exactly where things left off.

## Staging directory

```
data/
  metadata.parquet                              ← canonical (merged, complete)
  scene_graphs.parquet                          ← canonical
  _staging/
    scene_graphs_shard0_n3_b1744000000.parquet  ← pending (shard 0 of 3)
    scene_graphs_shard1_n3_b1744000000.parquet  ← pending (shard 1 of 3)
```

Any files in `_staging/` represent unmerged work. `merge.py` is idempotent and safe to run at any time.

---

## Usage

### Option A: SSH orchestration (automatic)

`distribute.py` SSHes to each machine, launches the appropriate shard, streams output with `[host]` prefixes, and exits non-zero if any machine fails.

```bash
# Download
uv run python scripts/distribute.py \
  --hosts user@gpu1 user@gpu2 user@gpu3 \
  --script download \
  --config configs/download.toml \
  --extra-args "--limit 20000"
uv run python scripts/merge.py --stage metadata

# Extract scene graphs (GPU required on each machine)
uv run python scripts/distribute.py \
  --hosts user@gpu1 user@gpu2 user@gpu3 \
  --script extract_scene_graphs \
  --concurrency 2
uv run python scripts/merge.py --stage scene_graphs

# Extract semantic annotations (API)
uv run python scripts/distribute.py \
  --hosts user@gpu1 user@gpu2 user@gpu3 \
  --script extract_semantic \
  --concurrency 16
uv run python scripts/merge.py --stage semantic_annotations

# Annotate
uv run python scripts/distribute.py \
  --hosts user@gpu1 user@gpu2 user@gpu3 \
  --script annotate \
  --extra-args "--pipeline semantic"
uv run python scripts/merge.py --stage annotated
```

`distribute.py` arguments:

| Flag | Description |
|---|---|
| `--hosts` | `user@host` strings, one per machine (sets `num_shards` automatically) |
| `--script` | One of: `download`, `extract_scene_graphs`, `extract_semantic`, `annotate` |
| `--config` | Path to config TOML (must be accessible on remote machines) |
| `--concurrency` | Workers per machine (passed as `--concurrency`) |
| `--extra-args` | Extra arguments forwarded verbatim to the script |
| `--project-dir` | Absolute path to `data-gen/` on remote machines (default: local path) |
| `--dry-run` | Print SSH commands without executing |

### Option B: Job scheduler (manual, e.g. SLURM)

Submit one job per shard using `--shard-id` and `--num-shards` directly.

```bash
# SLURM array job example (8 shards)
sbatch --array=0-7 --wrap="\
  uv run python scripts/extract_semantic.py \
    --shard-id \$SLURM_ARRAY_TASK_ID \
    --num-shards 8 \
    --concurrency 8"

# After all jobs complete:
uv run python scripts/merge.py --stage semantic_annotations
```

The `--shard-id` / `--num-shards` flags are available on all four pipeline scripts:
- `scripts/download.py`
- `scripts/extract_scene_graphs.py`
- `scripts/extract_semantic.py`
- `scripts/annotate.py`

### Option C: Single machine (unchanged behavior)

Omit `--shard-id` / `--num-shards` (defaults to `0`/`1`). Results write directly to the canonical parquet — no staging, no merge step needed.

```bash
uv run python scripts/extract_semantic.py --concurrency 16
```

---

## Merging

```bash
# Merge a single stage
uv run python scripts/merge.py --stage scene_graphs

# Merge all stages in pipeline order
uv run python scripts/merge.py --all

# Specify a custom data directory
uv run python scripts/merge.py --stage annotated --data-dir /path/to/data
```

`merge.py` is safe to run at any time:
- If no staging files exist it exits cleanly (no-op).
- It deduplicates by `filename` (or `source_url` for metadata), keeping the newest record.
- It writes atomically (temp file → rename) before deleting staging files.

---

## Failure recovery

If a machine fails mid-run:

1. Run `merge.py` to salvage work from completed shards.
2. Optionally change the number of machines.
3. Re-run the same command — each machine reads the canonical file, skips already-done rows, and processes only its remaining share.

**Example**: 3-machine download, machine 2 fails, extend to 20 000 rows with 4 machines:

```bash
# Initial run (3 machines, machine 2 dies)
uv run python scripts/distribute.py --hosts u@m0 u@m1 u@m2 --script download --extra-args "--limit 10000"

# Salvage completed shards
uv run python scripts/merge.py --stage metadata   # canonical now has ~6700 rows

# Re-run with 4 machines and extended limit
uv run python scripts/distribute.py --hosts u@m0 u@m1 u@m2 u@m3 --script download --extra-args "--limit 20000"
uv run python scripts/merge.py --stage metadata   # canonical now has all 20000 rows
```

---

## Checking for pending work

```bash
ls data/_staging/                        # non-empty = unmerged staging files exist
uv run python scripts/merge.py --all    # merge everything pending
```
