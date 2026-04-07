#!/usr/bin/env python
"""SSH-based distributed job launcher.

Launches a pipeline script on multiple remote machines in parallel, assigning
each machine a unique shard of the work. Prefixes each machine's output with
its hostname for easy log filtering.

For job schedulers (SLURM, PBS, etc.) you don't need this script — submit
jobs directly with --shard-id N --num-shards M on each node.

Usage:
    uv run python scripts/distribute.py \\
        --hosts user@gpu1 user@gpu2 user@gpu3 \\
        --script extract_scene_graphs \\
        [--config config.toml] \\
        [--concurrency 2] \\
        [--extra-args "--limit 20000 --pipeline semantic"] \\
        [--project-dir /path/to/data-gen] \\
        [--dry-run]

After all shards complete, merge results:
    uv run python scripts/merge.py --stage scene_graphs
"""

import argparse
import subprocess
import sys
import threading
from pathlib import Path

VALID_SCRIPTS = ["download", "extract_scene_graphs", "extract_semantic", "annotate"]


def stream_output(proc: subprocess.Popen, prefix: str, lock: threading.Lock) -> None:
    """Read proc stdout line-by-line and write with a [host] prefix."""
    for line in proc.stdout:
        with lock:
            sys.stdout.write(f"[{prefix}] {line.decode(errors='replace')}")
            sys.stdout.flush()


def build_ssh_cmd(
    host: str,
    script: str,
    shard_id: int,
    num_shards: int,
    project_dir: str,
    config: str | None,
    concurrency: int | None,
    extra_args: str,
) -> str:
    parts = [f"cd {project_dir}"]
    run_parts = ["uv run python", f"scripts/{script}.py"]
    run_parts += ["--shard-id", str(shard_id), "--num-shards", str(num_shards)]
    if config:
        run_parts += ["--config", config]
    if concurrency is not None:
        run_parts += ["--concurrency", str(concurrency)]
    if extra_args:
        run_parts.append(extra_args)
    parts.append(" ".join(run_parts))
    remote_cmd = " && ".join(parts)
    return f"ssh -o StrictHostKeyChecking=no {host} '{remote_cmd}'"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch distributed pipeline jobs via SSH")
    parser.add_argument(
        "--hosts",
        nargs="+",
        required=True,
        metavar="USER@HOST",
        help="Remote hosts to distribute work across (one shard per host)",
    )
    parser.add_argument(
        "--script",
        required=True,
        choices=VALID_SCRIPTS,
        help="Pipeline script to run on each machine",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config TOML (must be accessible on remote machines; default: config.toml)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of workers per machine (passed as --concurrency)",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        metavar="ARGS",
        help="Additional arguments forwarded verbatim to the script (quote the whole string)",
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).parent.parent.resolve()),
        help="Absolute path to the data-gen project root on remote machines (default: local cwd)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SSH commands without executing them",
    )
    args = parser.parse_args()

    hosts = args.hosts
    num_shards = len(hosts)

    cmds = [
        build_ssh_cmd(
            host=host,
            script=args.script,
            shard_id=shard_id,
            num_shards=num_shards,
            project_dir=args.project_dir,
            config=args.config,
            concurrency=args.concurrency,
            extra_args=args.extra_args,
        )
        for shard_id, host in enumerate(hosts)
    ]

    if args.dry_run:
        print(f"Would launch {num_shards} shard(s):")
        for cmd in cmds:
            print(f"  {cmd}")
        return

    print(f"Launching {num_shards} shard(s) across: {', '.join(hosts)}")

    lock = threading.Lock()
    procs: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []

    for host, cmd in zip(hosts, cmds):
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        t = threading.Thread(target=stream_output, args=(proc, host, lock), daemon=True)
        t.start()
        procs.append(proc)
        threads.append(t)

    for proc, t in zip(procs, threads):
        proc.wait()
        t.join()

    failed = [host for host, proc in zip(hosts, procs) if proc.returncode != 0]
    if failed:
        print(f"\nFAILED on: {', '.join(failed)}", file=sys.stderr)
        print("Run merge.py to salvage completed shards, then re-run distribute.py.", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll {num_shards} shards completed successfully.")
    print(f"Next: uv run python scripts/merge.py --stage {_stage_for(args.script)}")


def _stage_for(script: str) -> str:
    return {
        "download": "metadata",
        "extract_scene_graphs": "scene_graphs",
        "extract_semantic": "semantic_annotations",
        "annotate": "annotated",
    }[script]


if __name__ == "__main__":
    main()
