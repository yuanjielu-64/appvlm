#!/usr/bin/env python3
"""Count how many actor_* folders contain at least one actor_*.json file."""

import argparse
from pathlib import Path


def count_actor_dirs(root: Path) -> int:
    # Match directories named actor_* that contain any actor_*.json file
    total = 0
    for path in root.iterdir():
        if not (path.is_dir() and path.name.startswith("actor_")):
            continue
        has_actor_json = any(p.name.startswith("actor_") and p.suffix == ".json" for p in path.iterdir())
        if has_actor_json:
            total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count actor_* directories that include any actor_*.json file"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory)",
    )
    args = parser.parse_args()

    root_path = Path(args.root).expanduser().resolve()
    if not root_path.is_dir():
        raise SystemExit(f"Not a directory: {root_path}")

    total = count_actor_dirs(root_path)
    print(total)


if __name__ == "__main__":
    main()
