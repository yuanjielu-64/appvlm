#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

# 复用你现有的转换函数
from csv_to_json import csv_to_json

def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV->JSON for a range of actor folders (start inclusive, end exclusive)."
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="根目录，包含 actor_0, actor_1, ... 子目录，例如 /scratch/bwang25/appvlm/buffer/dwa_heurstic"
    )
    parser.add_argument(
        "--start", type=int, required=True,
        help="起始 actor index（包含），例如 0"
    )
    parser.add_argument(
        "--end", type=int, required=True,
        help="结束 actor index（不包含），例如 10 表示 actor_0..actor_9"
    )
    parser.add_argument(
        "--csv-name", default="data.csv",
        help="每个 actor 目录下的 CSV 文件名，默认 data.csv"
    )
    parser.add_argument(
        "--output-name", default=None,
        help="输出 JSON 文件名，默认用 actor_X.json；可指定固定文件名"
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        sys.exit(f"Root dir not found: {root}")

    for idx in range(args.start, args.end):
        actor_dir = root / f"actor_{idx}"
        if not actor_dir.is_dir():
            print(f"[SKIP] missing folder: {actor_dir}")
            continue

        csv_path = actor_dir / args.csv_name
        if not csv_path.exists():
            print(f"[SKIP] CSV not found: {csv_path}")
            continue

        out_name = args.output_name or f"{actor_dir.name}.json"
        out_path = actor_dir / out_name

        try:
            csv_to_json(str(csv_path), str(out_path))
        except Exception as e:
            print(f"[FAIL] {csv_path} -> {out_path}: {e}")
        else:
            print(f"[OK] {csv_path} -> {out_path}")

if __name__ == "__main__":
    main()
