#!/usr/bin/env python3
"""
Analyze nav_metric across data_trajectory.csv files and produce difficulty tiers.

Features:
- Recursively scans a root for actor_* folders (or any folders) containing data_trajectory.csv
- Checks whether nav_metric values are degenerate (e.g., all ~= 0.5)
- Aggregates nav_metric by group (world_id if present, else actor folder)
- Splits groups into clean/mid/hard by quantiles (e.g., top 20% clean, bottom 20% hard)
- Writes a difficulty_map.csv and prints a concise summary

Usage:
  python analyze_nav_metric.py --root /path/to/root \
    --trajectory-name data_trajectory.csv \
    --group-by actor --q-clean 0.8 --q-hard 0.2 \
    --out difficulty_map.csv
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


def find_trajectory_files(root: Path, trajectory_name: str) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob(trajectory_name):
        if p.is_file():
            print(p)
            files.append(p)
    return files


def load_df(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

def summarize_group(values: pd.Series) -> Dict[str, float]:
    v = values.dropna().astype(float)
    if v.empty:
        return {"count": 0, "mean": math.nan, "median": math.nan, "std": math.nan}
    return {
        "count": int(v.shape[0]),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "std": float(v.std(ddof=0)),
    }


def summarize_group_with_time(nav_values: pd.Series, time_values: pd.Series) -> Dict[str, float]:
    """Compute summary stats for nav_metric and average Time."""
    nav = nav_values.dropna().astype(float)
    time = time_values.dropna().astype(float)

    result = {
        "count": int(nav.shape[0]) if not nav.empty else 0,
        "nav_mean": float(nav.mean()) if not nav.empty else math.nan,
        "nav_median": float(nav.median()) if not nav.empty else math.nan,
        "nav_std": float(nav.std(ddof=0)) if not nav.empty else math.nan,
        "time_mean": float(time.mean()) if not time.empty else math.nan,
    }
    return result


def infer_actor_key(file_path: Path) -> str:
    """Infer actor folder name for a given trajectory file.

    If data_trajectory.csv is nested (e.g., actor_x/world_123/data_trajectory.csv),
    walk up parents to find the nearest directory whose name starts with 'actor_'.
    Fallback to immediate parent name if none matches.
    """
    for p in [file_path.parent] + list(file_path.parents):
        try:
            if p.name.startswith("actor_"):
                return p.name
        except Exception:
            continue
    return file_path.parent.name


def compute_difficulty(
    nav_groups: Dict[str, pd.Series],
    time_groups: Dict[str, pd.Series],
    q_clean: float = None,
    q_hard: float = None,
    quantiles: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute per-group mean(nav_metric), mean(Time), count, percent_vs_half.

    Returns a DataFrame with columns: key, count, score, avg_time, percent_vs_half
    """
    rows: List[Tuple[str, int, float, float]] = []
    for k, nav_s in nav_groups.items():
        v = nav_s.dropna().astype(float)
        if v.empty:
            continue

        nav_mean = float(v.mean())

        # Get corresponding time series if available
        time_s = time_groups.get(k, pd.Series(dtype=float))
        t = time_s.dropna().astype(float)
        time_mean = float(t.mean()) if not t.empty else math.nan

        rows.append((str(k), int(v.shape[0]), nav_mean, time_mean))

    df = pd.DataFrame(rows, columns=["key", "count", "score", "avg_time"]).dropna(subset=["score"])
    if df.empty:
        return df

    # Percent vs 0.5 baseline
    df["percent_vs_half"] = (df["score"] / 0.5) * 100.0

    # No tier calculation - just return the metrics
    return df

def main():
    ap = argparse.ArgumentParser(description="Analyze nav_metric and compute statistics (score, avg_time, count)")
    ap.add_argument("--root", type=Path, default="../../../ros_jackal/buffer/mppi_heurstic/", help="Root directory to scan (default: current working directory)")
    ap.add_argument("--trajectory-name", default="data_trajectory.csv", help="Trajectory CSV filename to search for")
    ap.add_argument(
        "--group-by",
        default="actor",
        choices=["auto", "world", "actor"],
        help=(
            "Grouping key: 'actor' (per data_trajectory.csv folder) by default; "
            "'world' requires a world_id column; 'auto' prefers 'world' if available."
        ),
    )
    ap.add_argument("--verbose", action="store_true", help="Print progress during scanning and aggregation")
    ap.add_argument("--progress-interval", type=int, default=50, help="How often (in files) to print progress when verbose")
    ap.add_argument("--out", type=Path, default=None, help="Where to write difficulty_map.csv; default under root")
    ap.add_argument("--trim-top", type=float, default=0.05, help="Per-file: drop the top fraction (e.g., 0.10) of nav_metric before aggregation")
    ap.add_argument("--trim-bottom", type=float, default=0.05, help="Per-file: drop the bottom fraction (e.g., 0.10) of nav_metric before aggregation")
    args = ap.parse_args()

    files = find_trajectory_files(args.root, args.trajectory_name)
    if not files:
        print(f"No {args.trajectory_name} found under {args.root}")
        return
    total_files = len(files)
    print(f"Found {total_files} '{args.trajectory_name}' files under {args.root}")

    # Gather series per group (both nav_metric and Time)
    nav_groups: Dict[str, pd.Series] = {}
    time_groups: Dict[str, pd.Series] = {}
    degenerate_files: List[Path] = []

    for i, f in enumerate(sorted(files), start=1):
        if args.verbose and (i == 1 or i % args.progress_interval == 0 or i == total_files):
            print(f"[scan] {i}/{total_files}: {f}")
        df = load_df(f)
        if df is None or "nav_metric" not in df.columns:
            if args.verbose:
                print(f"[skip] missing or invalid nav_metric: {f}")
            continue

        keys = [k for k in ['Method', 'World', 'Start_frame_id'] if k in df.columns]
        if keys:
            df = df.drop_duplicates(subset=keys, keep='last').reset_index(drop=True)
        elif 'Start_frame_id' in df.columns:
            df = df.drop_duplicates(subset='Start_frame_id', keep='last').reset_index(drop=True)

        # Step 1: Sort by nav_metric (desc) and Time (asc)
        if 'Time' in df.columns:
            df_sorted = df.sort_values(by=['nav_metric', 'Time'], ascending=[False, True]).reset_index(drop=True)
        else:
            df_sorted = df.sort_values(by=['nav_metric'], ascending=[False]).reset_index(drop=True)

        # Step 2: Apply trimming based on sorted order
        if args.trim_top > 0.0 or args.trim_bottom > 0.0:
            top = max(0.0, min(0.49, float(args.trim_top)))
            bot = max(0.0, min(0.49, float(args.trim_bottom)))
            n = int(df_sorted.shape[0])
            n_top = int(n * top)
            n_bot = int(n * bot)
            keep_start = n_top
            keep_end = max(keep_start, n - n_bot)
            if keep_start < keep_end:
                if args.verbose:
                    print(f"[trim] {f}: drop top={n_top}, bottom={n_bot}, keep {keep_end - keep_start}/{n}")
                df_sorted = df_sorted.iloc[keep_start:keep_end]
            else:
                if args.verbose:
                    print(f"[trim] {f}: skip trimming (requested drops >= total)")

        # Step 3: Determine key (world identifier from actor folder name)
        key = infer_actor_key(f)  # e.g., "actor_0" for world_0

        # Step 4: Aggregate nav_metric and Time
        # Filter out failed trajectories (nav_metric = 0) for Time calculation
        nav_s = df_sorted["nav_metric"].astype(float)
        nav_groups[key] = pd.concat([nav_groups.get(key, pd.Series(dtype=float)), nav_s], ignore_index=True)

        if 'Time' in df_sorted.columns:
            # Only include Time values where nav_metric > 0 (successful trajectories)
            successful_mask = df_sorted["nav_metric"].astype(float) > 0
            time_s = df_sorted.loc[successful_mask, "Time"].astype(float)

            if not time_s.empty:
                time_groups[key] = pd.concat([time_groups.get(key, pd.Series(dtype=float)), time_s], ignore_index=True)

    if not nav_groups:
        print("No valid nav_metric found to analyze.")
        return

    # Compute metrics (DataFrame)
    df_out = compute_difficulty(nav_groups, time_groups)
    if df_out.empty:
        print("No metrics computed (insufficient data).")
        return

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total actors analyzed: {len(df_out)}")
    print(f"\nScore (nav_metric):")
    print(f"  Min:    {df_out['score'].min():.4f}")
    print(f"  Max:    {df_out['score'].max():.4f}")
    print(f"  Mean:   {df_out['score'].mean():.4f}")
    print(f"  Median: {df_out['score'].median():.4f}")
    print(f"\nAverage Time (seconds):")
    print(f"  Min:    {df_out['avg_time'].min():.2f}s")
    print(f"  Max:    {df_out['avg_time'].max():.2f}s")
    print(f"  Mean:   {df_out['avg_time'].mean():.2f}s")
    print(f"  Median: {df_out['avg_time'].median():.2f}s")
    print("="*60 + "\n")

    # Write CSV
    out_path = args.out or (args.root / "difficulty_map.csv")
    # Ensure suffix
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")
    # Sort by score (descending) - best performers first
    df_out.sort_values("score", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote metrics CSV to {out_path}")


if __name__ == "__main__":
    main()
