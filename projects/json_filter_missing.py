#!/usr/bin/env python3
"""
Filter a JSON dataset by dropping samples whose image file is missing on disk.

Usage:
  python json_filter_missing.py \
    --data /scratch/bwang25/appvlm/buffer/dwa_heurstic/splits_200k/chunk_001.json \
    --image-root /scratch/bwang25/appvlm/buffer/dwa_heurstic \
    --out /scratch/bwang25/appvlm/buffer/dwa_heurstic/splits_200k/chunk_001.filtered.json
"""
import argparse
import json
from pathlib import Path


def filter_missing(data_path: Path, image_root: Path, out_path: Path) -> None:
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    kept = []
    missing = []
    for sample in data:
        image_rel = sample.get("image")
        if not image_rel:
            missing.append(("no_image_field", sample.get("id")))
            continue
        image_abs = image_root / image_rel
        if image_abs.exists():
            kept.append(sample)
        else:
            missing.append((image_rel, sample.get("id")))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    print(f"Loaded {len(data)} samples from {data_path}")
    print(f"Kept   {len(kept)} samples")
    print(f"Dropped {len(missing)} missing images")
    if missing:
        print("First few missing:", missing[:10])


def main():
    parser = argparse.ArgumentParser(description="Filter out samples with missing images.")
    parser.add_argument("--data", required=True, type=Path, help="Input JSON path")
    parser.add_argument("--image-root", required=True, type=Path, help="Root dir for images")
    parser.add_argument(
        "--out",
        type=Path,
        help="Output JSON path (default: <input>.filtered.json)",
    )
    args = parser.parse_args()

    out_path = args.out or args.data.with_suffix(".filtered.json")
    filter_missing(args.data, args.image_root, out_path)


if __name__ == "__main__":
    main()
