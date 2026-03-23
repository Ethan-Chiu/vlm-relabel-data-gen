#!/usr/bin/env python
"""Inspect a Parquet file.

Usage:
    uv run python scripts/show_parquet.py                                         # print table (default: data/relabeled.parquet)
    uv run python scripts/show_parquet.py data/metadata.parquet                   # different file
    uv run python scripts/show_parquet.py --cols filename new_caption             # specific columns
    uv run python scripts/show_parquet.py --head 20                               # first 20 rows
    uv run python scripts/show_parquet.py --info                                  # schema + row count
    uv run python scripts/show_parquet.py --pdf report.pdf                        # image + caption PDF
    uv run python scripts/show_parquet.py --pdf report.pdf --image-dir data/raw   # custom image directory
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _build_pdf(df: pd.DataFrame, image_dir: Path, out_path: Path) -> None:
    from fpdf import FPDF
    from PIL import Image

    # Page layout constants
    MARGIN = 12
    PAGE_W = 210                        # A4 mm
    IMG_MAX_W = PAGE_W - 2 * MARGIN
    IMG_MAX_H = 80
    LABEL_COLOR = (80, 80, 80)
    CAPTION_COLOR = (30, 30, 30)

    # All caption columns in display order — render whichever exist in the file
    CAPTION_COLUMNS: list[tuple[str, str]] = [
        ("caption",          "Original caption"),
        ("new_caption",      "Generated caption"),
        ("type_a",           "Type A — Spatial scene description"),
        ("type_b",           "Type B — Object-centric referring"),
        ("type_c",           "Type C — Action-conditioned"),
        ("spatial_caption",  "Spatial caption (two-call)"),
        ("scene_graph",      "Scene graph"),
        ("scene_type_a",     "Scene Type A — Grounded spatial description"),
        ("scene_type_b",     "Scene Type B — Grounded referring expressions"),
        ("scene_type_c",     "Scene Type C — Grounded action caption"),
    ]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=MARGIN)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    _fonts = Path("/System/Library/Fonts/Supplemental")
    pdf.add_font("Arial", style="",   fname=str(_fonts / "Arial.ttf"))
    pdf.add_font("Arial", style="B",  fname=str(_fonts / "Arial Bold.ttf"))
    pdf.add_font("Arial", style="I",  fname=str(_fonts / "Arial Italic.ttf"))
    pdf.add_font("Arial", style="BI", fname=str(_fonts / "Arial Bold Italic.ttf"))

    missing = 0
    for _, row in df.iterrows():
        img_path = image_dir / row["filename"]
        if not img_path.exists():
            missing += 1
            continue

        pdf.add_page()

        # --- Image (scaled to fit, centered) ---
        with Image.open(img_path) as im:
            iw, ih = im.size
        scale = min(IMG_MAX_W / iw, IMG_MAX_H / ih)
        draw_w, draw_h = iw * scale, ih * scale
        pdf.image(str(img_path), x=MARGIN + (IMG_MAX_W - draw_w) / 2, y=MARGIN, w=draw_w, h=draw_h)

        # --- Filename ---
        pdf.set_xy(MARGIN, MARGIN + draw_h + 6)
        pdf.set_font("Arial", "B", 9)
        pdf.set_text_color(*LABEL_COLOR)
        pdf.cell(0, 5, row["filename"])
        pdf.ln(8)

        # --- Caption sections (whichever columns are present) ---
        for col, label in CAPTION_COLUMNS:
            if col not in row or pd.isna(row[col]):
                continue
            pdf.set_font("Arial", "BI", 8)
            pdf.set_text_color(*LABEL_COLOR)
            pdf.cell(0, 4, label)
            pdf.ln(5)
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(*CAPTION_COLOR)
            pdf.multi_cell(IMG_MAX_W, 5, str(row[col]))
            pdf.ln(3)

    pdf.output(str(out_path))

    n = len(df) - missing
    present = [label for col, label in CAPTION_COLUMNS if col in df.columns]
    print(f"PDF written → {out_path}  ({n} images, {missing} missing)")
    print(f"Sections:  {', '.join(present)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Parquet file")
    parser.add_argument("path", nargs="?", default="data/relabeled.parquet", help="Path to Parquet file")
    parser.add_argument("--cols", nargs="+", help="Only show these columns")
    parser.add_argument("--head", type=int, help="Show only first N rows")
    parser.add_argument("--info", action="store_true", help="Show schema and memory usage instead of rows")
    parser.add_argument("--pdf", metavar="OUTPUT", help="Generate a PDF report with images and captions")
    parser.add_argument("--image-dir", default="data/raw", help="Directory containing images (default: data/raw)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(path)

    if args.head:
        df = df.head(args.head)

    # --- PDF mode ---
    if args.pdf:
        _build_pdf(df, image_dir=Path(args.image_dir), out_path=Path(args.pdf))
        return

    # --- Terminal mode ---
    if args.info:
        print(f"File:    {path}")
        print(f"Rows:    {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print()
        df.info()
        return

    if args.cols:
        missing = [c for c in args.cols if c not in df.columns]
        if missing:
            print(f"Unknown columns: {missing}. Available: {df.columns.tolist()}", file=sys.stderr)
            sys.exit(1)
        df = df[args.cols]

    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.max_rows", None)
    print(df.to_string())


if __name__ == "__main__":
    main()
