"""Download images from a list of annotations and build a metadata index."""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import requests
from loguru import logger
from PIL import Image, ImageFile, UnidentifiedImageError

from datagen.config import Config
from datagen.storage import read_metadata, write_metadata

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

_KNOWN_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


class Annotation(TypedDict):
    caption: str
    url: str


def run(anns: list[Annotation], cfg: Config, output_path: Path | None = None) -> None:
    """Download annotations and write metadata.

    Each downloaded image is validated (PIL-loadable, meets min_image_pixels).
    All records are written to metadata.parquet with a ``valid`` boolean field.
    Images that fail validity are additionally recorded in filtered_download.parquet
    with their filter_reason (kept separate to avoid cluttering metadata.parquet).

    output_path: if set, write results here instead of cfg.metadata_path.
                 Skip logic always reads cfg.metadata_path (the canonical file)
                 so changing the output path does not affect resume behavior.
    """
    out = output_path or cfg.metadata_path
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip URLs already present in the canonical metadata index.
    already_done: set[str] = set()
    if cfg.metadata_path.exists():
        existing = read_metadata(cfg.metadata_path)
        already_done = set(existing["source_url"].tolist())
        logger.info(f"Resuming — {len(already_done)} already downloaded, skipping those")

    pending_count = sum(1 for a in anns if a["url"] not in already_done)
    stats = {
        "total": pending_count,
        "success": 0,
        "invalid": 0,   # downloaded but failed validity check
        "failed": 0,    # could not download at all
        "skipped": len(already_done),
    }
    error_counts: dict[str, int] = defaultdict(int)
    records: list[dict] = []
    invalid_entries: list[dict] = []   # {filename, filter_reason} for filtered_download.parquet

    for i, ann in enumerate(anns):
        url = ann["url"]
        if url in already_done:
            continue

        # ── Step 1: fetch bytes ────────────────────────────────────────────────
        try:
            response = requests.get(url, timeout=cfg.timeout)
            response.raise_for_status()
            data = response.content
        except requests.exceptions.Timeout:
            error_counts["Timeout"] += 1
            stats["failed"] += 1
            continue
        except requests.exceptions.HTTPError as e:
            error_counts[f"HTTP {e.response.status_code}"] += 1
            stats["failed"] += 1
            continue
        except requests.exceptions.RequestException as e:
            error_counts[f"RequestError({type(e).__name__})"] += 1
            stats["failed"] += 1
            continue

        # ── Step 2: determine filename ─────────────────────────────────────────
        suffix = Path(url.split("?")[0]).suffix.lower()
        if suffix not in _KNOWN_SUFFIXES:
            suffix = ".jpg"
        filename = f"{i:06d}{suffix}"

        # ── Step 3: parse image and run validity checks ────────────────────────
        valid = True
        filter_reason: str | None = None
        w = h = 0
        mode = ""
        image: Image.Image | None = None

        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            w, h = image.width, image.height
            mode = image.mode
            if w * h < cfg.min_image_pixels:
                valid = False
                filter_reason = (
                    f"too_small: {w}×{h}={w * h} px (min {cfg.min_image_pixels})"
                )
        except (UnidentifiedImageError, OSError, Exception) as exc:
            valid = False
            filter_reason = f"load_error: {str(exc)[:120]}"

        # ── Step 4: save file ──────────────────────────────────────────────────
        img_path = cfg.output_dir / filename
        if image is not None:
            image.save(img_path)
        else:
            # PIL couldn't parse it — save raw bytes so the file exists on disk
            img_path.write_bytes(data)

        # ── Step 5: record ─────────────────────────────────────────────────────
        record = {
            "filename": filename,
            "caption": ann["caption"],
            "source_url": url,
            "width": w,
            "height": h,
            "mode": mode,
            "valid": valid,
        }
        records.append(record)

        if valid:
            stats["success"] += 1
            logger.info(f"[OK]      {w}×{h} → {filename}  {ann['caption'][:60]}")
        else:
            stats["invalid"] += 1
            invalid_entries.append({"filename": filename, "filter_reason": filter_reason})
            logger.debug(f"[INVALID] {filename}: {filter_reason}")

    # ── Flush results ──────────────────────────────────────────────────────────
    write_metadata(records, out)

    if invalid_entries:
        cfg.filtered_download_path.parent.mkdir(parents=True, exist_ok=True)
        write_metadata(invalid_entries, cfg.filtered_download_path)
        logger.info(
            f"{stats['invalid']} invalid images recorded → {cfg.filtered_download_path}"
        )

    logger.info(
        f"Done: {stats['success']} valid, {stats['invalid']} invalid, "
        f"{stats['failed']} failed to download, {stats['skipped']} skipped"
        f" — index → {out}"
    )
    if error_counts:
        logger.warning(
            "Download errors: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(error_counts.items()))
        )
