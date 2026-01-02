import csv
import json
import os.path
from pathlib import Path
import argparse
import pandas as pd

from utils import get_row_from_trajectory
import numpy as np
from typing import Dict, Any, List
import math

SYSTEM_PROMPT = """
    You are a navigation scene analyzer for Clearpath Jackal robot motion planning.
    Scene Representation (Costmap Visualization):
    - Scale: Each grid cell represents approximately 1 meter.
    - Robot: Yellow square (0.508 m x 0.430 m footprint)
    - Green arrow: forward driving direction (x-axis)
    - Blue line: lateral direction (y-axis)
    - Obstacles: Red points from laser scanner measurements
    - Global Path: Black curve showing a kinematic reference trajectory that guides direction but does not encode dynamics or feasibility.
    - Background: Grey grid is for visualization only and has no obstacle or cost meaning.
    Your Role:
    Analyze the spatial layout to understand:
    - Obstacle distribution and density around the robot
    - Available free space and safe clearance margins
    - Path geometry (straight corridor, sharp turn, narrow passage)
    - Motion feasibility given the current velocity state
    
    Your navigation reasoning will be used downstream to select parameters for the local motion planner.

"""

USER_PROMPT = """
    Current robot state:
    - Linear velocity: {linear_vel:.3f} m/s
    - Angular velocity: {angular_vel:.3f} rad/s
    
    Target local planner: {algorithm}
    
    Use the scene image and robot state to perform navigation reasoning about obstacle proximity, path curvature, free-space structure, and motion constraints.
"""

ALGORITHM_PARAMS = {
    "DWA": {
        "max_vel_x":        {"range": [0.2, 2.0], "default": 1.5,  "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_theta":    {"range": [0.314, 3.14], "default": 1.57, "type": "float", "description": "Angular velocity (rad/s)"},
        "vx_samples":       {"range": [4, 12], "default": 6,   "type": "int",   "description": "Linear velocity samples"},
        "vtheta_samples":   {"range": [8, 40], "default": 20,  "type": "int",   "description": "Angular velocity samples"},
        "path_distance_bias": {"range": [0.1, 1.5], "default": 0.75, "type": "float", "description": "Path following weight"},
        "goal_distance_bias": {"range": [0.1, 2.0], "default": 1.0,  "type": "float", "description": "Goal seeking weight"},
        "inflation_radius": {"range": [0.1, 0.6], "default": 0.3, "type": "float", "description": "Inflation radius (m)"}
    },

    "TEB": {
        "max_vel_x":        {"range": [0.2, 2.0], "default": 2.0, "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_x_backwards": {"range": [0.1, 0.7], "default": 0.5, "type": "float", "description": "Backward velocity (m/s)"},
        "max_vel_theta":    {"range": [0.314, 3.14], "default": 3.0, "type": "float", "description": "Angular velocity (rad/s)"},
        "dt_ref":           {"range": [0.1, 0.35], "default": 0.25, "type": "float", "description": "Temporal resolution (s)"},
        "min_obstacle_dist": {"range": [0.05, 0.2], "default": 0.15, "type": "float", "description": "Minimum obstacle distance (m)"},
        "inflation_dist":   {"range": [0.01, 0.2], "default": 0.25, "type": "float", "description": "Inflation distance (m)"},
        "inflation_radius": {"range": [0.1, 0.6], "default": 0.2,  "type": "float", "description": "Inflation radius (m)"}
    },

    "MPPI": {
        "max_vel_x":        {"range": [-0.5, 2.0], "default": 1.5, "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_theta":    {"range": [0.314, 3.14], "default": 2.0, "type": "float", "description": "Angular velocity (rad/s)"},
        "nr_pairs_":         {"range": [400, 800], "default": 600, "type": "int", "description": "Rollout pairs"},
        "nr_steps_":         {"range": [20, 40], "default": 20, "type": "int", "description": "Prediction steps"},
        "linear_stddev":    {"range": [0.05, 0.15], "default": 0.1, "type": "float", "description": "Linear stddev"},
        "angular_stddev":   {"range": [0.02, 0.15], "default": 0.05, "type": "float", "description": "Angular stddev"},
        "lambda":           {"range": [0.5, 5.0], "default": 1.0, "type": "float", "description": "Temperature parameter"},
        "inflation_radius": {"range": [0.1, 0.6], "default": 0.25, "type": "float", "description": "Inflation radius (m)"}
    },

    "DDP": {
        "max_vel_x":        {"range": [0.0, 2.0], "default": 1.5, "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_theta":    {"range": [0.314, 3.14], "default": 3.0, "type": "float", "description": "Angular velocity (rad/s)"},
        "nr_pairs_":         {"range": [400, 800], "default": 600, "type": "int", "description": "Rollout pairs"},
        "distance":         {"range": [0.01, 0.2], "default": 0.1, "type": "float", "description": "Distance threshold (m)"},
        "robot_radius":     {"range": [0.01, 0.05], "default": 0.02, "type": "float", "description": "Robot radius (m)"},
        "inflation_radius": {"range": [0.1, 0.6], "default": 0.25, "type": "float", "description": "Inflation radius (m)"}
    }
}

# ------------------------------
# Helpers for defaults/prev params
# ------------------------------
def _default_from_config(param_info: Dict[str, Any]) -> float:
    """Return default value for a parameter.
    Priority: explicit 'default' in config; otherwise mid-point of range.
    Int types are rounded to nearest int within range.
    """
    if isinstance(param_info, dict) and 'default' in param_info:
        return param_info['default']
    lo, hi = param_info.get('range', [0.0, 1.0])
    mid = (float(lo) + float(hi)) / 2.0
    if param_info.get('type') == 'int':
        mid_i = int(round(mid))
        # clamp into range
        return max(int(lo), min(int(hi), mid_i))
    return mid

def _build_defaults(param_config: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    return {name: _default_from_config(info) for name, info in param_config.items()}

def _csv_column_name(param_name: str) -> str:
    """Map param_config key to actual CSV column name.
    inflation_radius -> final_inflation, others stay the same.
    """
    return 'final_inflation' if param_name == 'inflation_radius' else param_name

def _row_param_value(row: pd.Series, name: str, info: Dict[str, Any]):
    """Extract current row's parameter value by name; handle inflation mapping.
    Returns None if missing/NaN.
    """
    csv_name = _csv_column_name(name)
    if csv_name in row and pd.notna(row[csv_name]):
        v = row[csv_name]
        try:
            if info.get('type') == 'int':
                return int(v)
            return float(v)
        except Exception:
            return None
    return None

# =====================================================================================
def csv_to_json(input_csv_path, output_json_path, data_trajectory, param_config, alg, max_steps=None):
    data = []

    # Statistics counters
    stats = {
        'total_rows_processed': 0,
        'skipped_no_image': 0,
        'skipped_missing_params': 0,
        'valid_samples': 0,
        'trajectory_steps': []  # Track steps per trajectory
    }

    df_filtered, world_info = get_row_from_trajectory(data_trajectory, FILES, max_steps_per_trajectory=max_steps)

    # Skip if no trajectories after filtering
    if df_filtered.empty:
        print(f"[SKIP] No trajectories after filtering for {data_trajectory}")
        return

    csv_path = Path(input_csv_path)
    parent_dir = csv_path.parent
    actor_name = csv_path.parent.name

    df_full = pd.read_csv(input_csv_path)
    numeric_cols = df_full.select_dtypes(include=['float64', 'float32']).columns
    df_full[numeric_cols] = df_full[numeric_cols].round(4)
    df_full = df_full.drop_duplicates(subset=['Method', 'img_label'], keep='last')
    df_full = df_full.reset_index(drop=True)

    # ------------------------------
    # Compute previous-parameter columns for each row in df_full
    # - Reset to defaults at trajectory starts (Start_frame_id for this Method)
    # - Otherwise, previous params come from the immediately preceding row
    #   (sorted by img_label within Method)
    # ------------------------------
    # Initialize prev_* columns (using CSV column names)
    for pname in param_config.keys():
        csv_name = _csv_column_name(pname)
        df_full[f'prev_{csv_name}'] = math.nan

    defaults = _build_defaults(param_config)

    # Build a mapping of Method -> set(Start_frame_id)
    starts_by_method = {}
    if 'Method' in df_filtered.columns and 'Start_frame_id' in df_filtered.columns:
        for m, g in df_filtered.groupby('Method'):
            try:
                starts_by_method[m] = set(g['Start_frame_id'].astype(int).tolist())
            except Exception:
                starts_by_method[m] = set()
    else:
        # Fallback: no trajectory info; treat first row per Method as start
        starts_by_method = {m: set() for m in df_full['Method'].unique()}

    # For each Method, walk rows ordered by img_label
    for method, gidx in df_full.groupby('Method').groups.items():
        sub = df_full.loc[gidx].sort_values('img_label')
        prev_vals = defaults.copy()
        method_starts = starts_by_method.get(method, set())

        first = True
        for idx, row in sub.iterrows():
            img_label = None
            try:
                img_label = int(row.get('img_label', -1))
            except Exception:
                img_label = -1

            # Reset at beginning: either the very first element in this Method
            # or when hitting a declared Start_frame_id
            if first or (img_label in method_starts):
                prev_vals = defaults.copy()
                first = False

            # write prev_* columns for this row (using CSV column names)
            for pname in param_config.keys():
                csv_name = _csv_column_name(pname)
                df_full.at[idx, f'prev_{csv_name}'] = prev_vals[pname]

            # update prev with current row values for next row
            for pname, pinfo in param_config.items():
                cur = _row_param_value(row, pname, pinfo)
                if cur is not None:
                    prev_vals[pname] = cur

    grouped_filtered = df_filtered.groupby('Method')

    for method, group in grouped_filtered:

        df_method = df_full[df_full['Method'] == method].sort_values('img_label').reset_index(drop=True)

        for _, traj_row in group.iterrows():
            start = int(traj_row['Start_frame_id'])
            done = int(traj_row['Done_frame_id'])

            segment = df_method[(df_method['img_label'] >= start) & (df_method['img_label'] <= done)]
            num_steps = len(segment) - 1

            # Record trajectory step count
            stats['trajectory_steps'].append(num_steps)

            for idx in range(num_steps):
                stats['total_rows_processed'] += 1
                row = segment.iloc[idx]
                img_label = int(row["img_label"])
                sample_id = f"{method}_{img_label:06d}"

                image_filename = f"{method}_{img_label:06d}.png"
                image_rel_path = str(Path(actor_name) / image_filename)

                if not (parent_dir / image_filename).exists():
                    stats['skipped_no_image'] += 1
                    continue

                # Build user prompt (NO output constraints for DPT head)
                user_text = USER_PROMPT.format(
                    algorithm=alg,
                    linear_vel=row["linear_vel"],
                    angular_vel=row["angular_vel"]
                )
                human_value = "<image>\n" + user_text

                # Extract ground truth parameters (for regression labels)
                parameters = []
                valid_current = True
                for param_name, param_info in param_config.items():
                    csv_param_name = _csv_column_name(param_name)

                    if csv_param_name in row and pd.notna(row[csv_param_name]):
                        value = row[csv_param_name]
                        if param_info["type"] == "int":
                            value = int(value)
                        else:
                            value = float(value)
                        parameters.append(value)
                    else:
                        valid_current = False
                        break

                if not valid_current:
                    stats['skipped_missing_params'] += 1
                    continue

                # Extract previous parameters from precomputed columns
                prev_parameters: List[float] = []
                for param_name, param_info in param_config.items():
                    csv_name = _csv_column_name(param_name)
                    prev_col = f'prev_{csv_name}'
                    if prev_col in row and pd.notna(row[prev_col]):
                        v = row[prev_col]
                        if param_info["type"] == "int":
                            v = int(v)
                        else:
                            v = float(v)
                        prev_parameters.append(v)
                    else:
                        # fallback to defaults if missing
                        dv = _default_from_config(param_info)
                        if param_info["type"] == "int":
                            prev_parameters.append(int(dv))
                        else:
                            prev_parameters.append(float(dv))

                entry = {
                    "id": sample_id,
                    "images": [image_rel_path],  # Use "images" for regression collator
                    "parameters": parameters,     # Ground truth parameters (current)
                    "prev_parameters": prev_parameters,  # Previous parameters (same order)
                    "conversations": [
                        human_value,              # User prompt with <image> token
                    ],
                    "system_prompt": SYSTEM_PROMPT
                }
                data.append(entry)
                stats['valid_samples'] += 1

    # Remove existing output file before saving
    output_path = Path(output_json_path)
    if output_path.exists():
        output_path.unlink()
        print(f"[INFO] Removed existing file: {output_json_path}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Print statistics
    traj_steps = stats['trajectory_steps']

    print(f"\n{'='*80}")
    print(f"Conversion Statistics for {actor_name}")
    print(f"{'='*80}")
    print(f"Total rows processed:        {stats['total_rows_processed']}")
    print(f"Skipped (no image):          {stats['skipped_no_image']}")
    print(f"Skipped (missing params):    {stats['skipped_missing_params']}")
    print(f"Valid samples added:         {stats['valid_samples']}")
    print(f"{'-'*80}")
    print(f"Trajectory Statistics:")
    print(f"  Number of trajectories:    {len(traj_steps)}")
    if traj_steps:
        print(f"  Steps per trajectory:")
        print(f"    Mean:     {np.mean(traj_steps):.1f}")
        print(f"    Median:   {np.median(traj_steps):.1f}")
        print(f"    Std Dev:  {np.std(traj_steps):.1f}")
        print(f"    Min:      {np.min(traj_steps)}")
        print(f"    Max:      {np.max(traj_steps)}")
        print(f"    25th %ile: {np.percentile(traj_steps, 25):.1f}")
        print(f"    75th %ile: {np.percentile(traj_steps, 75):.1f}")
    print(f"{'='*80}")
    print(f"Output: {output_json_path}")
    print(f"{'='*80}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert CSV to JSON for VLM training')
    parser.add_argument('--alg', default="ddp")
    parser.add_argument('--root_dir', default="/home/yuanjielu/robot_navigation/noetic/app_data/ddp_heurstic")
    parser.add_argument('--csv_name', default="data.csv")
    parser.add_argument('--trajectory_name', default="data_trajectory.csv")
    parser.add_argument('--max_steps', type=int, default= 100 ,
                        help='Filter trajectories with more than this many steps (for data balancing)')
    args = parser.parse_args()

    alg_upper = args.alg.upper()

    if alg_upper not in ALGORITHM_PARAMS:
        raise ValueError(f"Unknown method: {args.alg}")

    param_config = ALGORITHM_PARAMS[alg_upper]

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    FILES = os.path.join(root_dir, "difficulty_map.csv")

    actor_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("actor_")])

    for actor_dir in actor_dirs:
        csv_files = [actor_dir / args.csv_name] if (actor_dir / args.csv_name).exists() else []
        data_files = actor_dir / args.trajectory_name

        if not csv_files or not data_files.exists():
            print(f"[SKIP] {actor_dir}")
            continue

        for csv_file in csv_files:
            output_json = actor_dir / f"{actor_dir.name}.json"
            csv_to_json(str(csv_file), str(output_json), str(data_files), param_config, alg_upper, max_steps=args.max_steps)
