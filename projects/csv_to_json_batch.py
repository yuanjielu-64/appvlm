#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

# 复用新版的转换函数
from csv_to_json_new import csv_to_json, ALGORITHM_PARAMS


def main():
    parser = argparse.ArgumentParser(
        description="Convert all actor_* folders under root from CSV to JSON (using trajectory sampling)."
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="根目录，包含 actor_0, actor_1, ... 子目录，例如 /scratch/bwang25/app_data/dwa_heurstic"
    )
    parser.add_argument(
        "--csv-name", default="data.csv",
        help="每个 actor 目录下的 CSV 文件名"
    )
    parser.add_argument(
        "--trajectory-name", default="data_trajectory.csv",
        help="每个 actor 目录下的轨迹文件名"
    )
    parser.add_argument(
        "--alg", default="teb",
        help="算法名称，对应 csv_to_json_new.ALGORITHM_PARAMS 的键"
    )
    parser.add_argument(
        "--output-name", default=None,
        help="输出 JSON 文件名，默认 actor_X.json"
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        sys.exit(f"Root dir not found: {root}")

    alg_upper = args.alg.upper()
    if alg_upper not in ALGORITHM_PARAMS:
        sys.exit(f"Unknown alg: {args.alg}; available: {', '.join(ALGORITHM_PARAMS.keys())}")

    param_config = ALGORITHM_PARAMS[alg_upper]

    actor_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("actor_")])
    if not actor_dirs:
        sys.exit(f"No actor_* folders under {root}")

    for actor_dir in actor_dirs:
        csv_path = actor_dir / args.csv_name
        traj_path = actor_dir / args.trajectory_name

        if not csv_path.exists() or not traj_path.exists():
            print(f"[SKIP] missing csv/trajectory in {actor_dir}")
            continue

        out_name = args.output_name or f"{actor_dir.name}.json"
        out_path = actor_dir / out_name

        try:
            csv_to_json(str(csv_path), str(out_path), str(traj_path), param_config, alg_upper)
        except Exception as e:
            print(f"[FAIL] {csv_path} -> {out_path}: {e}")
        else:
            print(f"[OK] {csv_path} -> {out_path}")


if __name__ == "__main__":
    main()
