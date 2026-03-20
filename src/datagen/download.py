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
from datagen.storage import write_metadata

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class Annotation(TypedDict):
    caption: str
    url: str


def download_image(url: str, timeout: int) -> Image.Image:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def run(anns: list[Annotation], cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(anns), "success": 0, "failed": 0}
    error_counts: dict[str, int] = defaultdict(int)
    records = []

    for i, ann in enumerate(anns):
        url = ann["url"]
        try:
            image = download_image(url, cfg.timeout)

            suffix = Path(url.split("?")[0]).suffix or ".jpg"
            filename = f"{i:06d}{suffix}"
            image.save(cfg.output_dir / filename)

            records.append({
                "filename": filename,
                "caption": ann["caption"],
                "source_url": url,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            })
            logger.info(f"[OK] {image.size} → {filename}  {ann['caption'][:60]}")
            stats["success"] += 1

        except requests.exceptions.Timeout:
            error_counts["Timeout"] += 1
            stats["failed"] += 1
        except requests.exceptions.HTTPError as e:
            error_counts[f"HTTP {e.response.status_code}"] += 1
            stats["failed"] += 1
        except requests.exceptions.RequestException as e:
            error_counts[f"RequestError({type(e).__name__})"] += 1
            stats["failed"] += 1
        except UnidentifiedImageError:
            error_counts["InvalidImage"] += 1
            stats["failed"] += 1

    write_metadata(records, cfg.metadata_path)

    logger.info(
        f"Done: {stats['success']}/{stats['total']} succeeded, "
        f"{stats['failed']} failed — index → {cfg.metadata_path}"
    )
    if error_counts:
        logger.warning(
            "Error breakdown: " + ", ".join(f"{k}: {v}" for k, v in sorted(error_counts.items()))
        )
