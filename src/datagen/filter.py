"""Image validity filter for the download pipeline.

check_image_validity() is called from download.py after the image file has been
saved to disk, to re-validate images that may have been partially written or
corrupted since the initial download.  During the normal download flow, validity
is checked inline (PIL image is already in memory), so this function is used for
re-validation scenarios only.

Scene complexity filtering (too few objects) is handled inline in scene_pipeline.py
using count_active_detections() from datagen.scene.geometry — no regex involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FilterResult:
    reason: str     # short code stored in filtered.parquet, e.g. "load_error"
    details: str    # human-readable description stored in filtered.parquet


def check_image_validity(img_path: Path, min_pixels: int) -> FilterResult | None:
    """Return FilterResult if the image is invalid, None if it passes all checks.

    Checks (in order):
      1. File exists on disk
      2. File size ≥ 512 bytes  — rejects empty stubs and HTTP error pages saved as files
      3. PIL can open the file and read dimensions  — catches truncated / corrupt images
      4. Width × height ≥ min_pixels  — rejects thumbnail-sized images

    PIL is imported lazily so this module can be imported without Pillow installed
    (e.g. in worker processes that don't run the prefilter).
    """
    from PIL import Image, UnidentifiedImageError

    if not img_path.exists():
        return FilterResult("missing_file", f"Not found: {img_path}")

    size = img_path.stat().st_size
    if size < 512:
        return FilterResult("file_too_small", f"{size} bytes (minimum 512)")

    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except (UnidentifiedImageError, OSError, Exception) as exc:
        return FilterResult("load_error", str(exc)[:200])

    if w == 0 or h == 0:
        return FilterResult("zero_dimension", f"{w}×{h} px")

    if w * h < min_pixels:
        return FilterResult(
            "too_small",
            f"{w}×{h} = {w * h} px (minimum {min_pixels})",
        )

    return None
