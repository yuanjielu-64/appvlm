#!/usr/bin/env python3
"""
Convert CSV trajectory data to APPLR pickle format.

This script reads data_trajectory.csv and data.csv files from actor directories,
extracts trajectories, converts costmap images to laser scans, and saves as pickle files.
"""

import pandas as pd
import pickle
import numpy as np
from PIL import Image
import os
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add path for planner_configs
sys.path.append(os.path.join(os.path.dirname(__file__), '../../lmms-finetune-qwen'))
from planner_configs import get_param_names


def extract_red_pixels_to_laserscan(image_path: str, num_beams: int = 721, laser_clip: float = 2.0) -> np.ndarray:
    """
    Extract red pixels from costmap image and convert to laser scan format using raycasting.

    Matches baseline APPLR normalization:
    - Raw range: 0-2 meters (clipped)
    - Normalized: (raw - 1.0) / 2.0 -> range [-0.5, 0.5]

    Args:
        image_path: Path to the costmap image
        num_beams: Number of laser scan beams (default 721 for 270° FOV)
        laser_clip: Maximum laser range in meters (default 2.0, same as baseline)

    Returns:
        laser_scan: Array of shape (1, 721) with normalized distances in range [-0.5, 0.5]
    """
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    pixels = np.array(img)

    # Extract red pixels (obstacles)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    red_mask = (red > 200) & (green < 100) & (blue < 100)

    # Find robot center (yellow square)
    yellow_mask = (red > 200) & (green > 200) & (blue < 100)
    if np.any(yellow_mask):
        robot_y, robot_x = np.where(yellow_mask)
        center_x = int(np.mean(robot_x))
        center_y = int(np.mean(robot_y))
    else:
        center_x = width // 2
        center_y = height // 2

    # Initialize laser scan with max range (in meters, before normalization)
    laser_scan_meters = np.full(num_beams, laser_clip, dtype=np.float64)

    # 270° FOV: from -135° to +135° relative to robot's forward direction
    fov_rad = np.deg2rad(270)
    start_angle = -fov_rad / 2

    # Raycast for each beam
    for beam_idx in range(num_beams):
        angle = start_angle + (beam_idx / (num_beams - 1)) * fov_rad
        dx_unit = np.cos(angle)
        dy_unit = -np.sin(angle)  # Image y-axis is down

        # Search along ray
        max_pixel_dist = int(np.sqrt(width**2 + height**2))
        for dist_pixels in range(1, max_pixel_dist):
            x = int(center_x + dx_unit * dist_pixels)
            y = int(center_y + dy_unit * dist_pixels)

            if x < 0 or x >= width or y < 0 or y >= height:
                break

            if red_mask[y, x]:
                # Convert pixel distance to meters
                # Assume costmap represents ~10m x 10m area (typical for local planning)
                pixels_per_meter = width / 10.0
                dist_meters = dist_pixels / pixels_per_meter

                # Clip to max range
                dist_meters = min(dist_meters, laser_clip)

                laser_scan_meters[beam_idx] = dist_meters
                break

    # Apply baseline normalization: (x - laser_clip/2) / laser_clip
    # This maps [0, laser_clip] -> [-0.5, 0.5]
    laser_scan_normalized = (laser_scan_meters - laser_clip / 2.0) / laser_clip

    return laser_scan_normalized.reshape(1, num_beams)


def extract_parameters_from_data_csv(data_df: pd.DataFrame, start_frame: int, done_frame: int,
                                      method: str, planner: str) -> List[np.ndarray]:
    """
    Extract parameters for each frame in the trajectory from data.csv.

    Args:
        data_df: DataFrame from data.csv
        start_frame: Starting frame ID
        done_frame: Ending frame ID
        method: Method name (e.g., 'RL', 'HB', 'DDP')
        planner: Planner name (e.g., 'dwa', 'teb', 'mppi', 'ddp')

    Returns:
        List of parameter arrays, one for each frame
    """
    param_names = get_param_names(planner)

    # CSV column name mapping to planner parameter names
    # data.csv uses different names than planner_configs
    csv_to_param_mapping = {
        'inflation_radius': 'final_inflation',  # Use final_inflation for inflation_radius
        'next_linear_vel': 'next_linear_vel:',   # Note the colon in CSV
    }

    params_list = []

    # Filter data for this trajectory
    traj_data = data_df[
        (data_df['Method'] == method) &
        (data_df['img_label'] >= start_frame) &
        (data_df['img_label'] <= done_frame)
    ].sort_values('img_label')

    for _, row in traj_data.iterrows():
        # Extract parameters based on planner config
        params = []
        for param_name in param_names:
            # Check if there's a mapping for this parameter
            csv_col = csv_to_param_mapping.get(param_name, param_name)

            if csv_col in row:
                params.append(row[csv_col])
            elif param_name in row:
                params.append(row[param_name])
            else:
                # Handle missing parameters
                print(f"Warning: Parameter {param_name} (CSV: {csv_col}) not found for {method}, using 0.0")
                params.append(0.0)

        # Remove last 2 parameters (next_linear_vel, next_angular_vel) as per your modification
        params = params[:-2]
        params_list.append(np.array(params, dtype=np.float32))

    return params_list


def convert_trajectory_to_pickle(
    traj_row: pd.Series,
    data_df: pd.DataFrame,
    actor_dir: str,
    planner: str
) -> List:
    """
    Convert one trajectory from CSV to APPLR pickle format.

    Args:
        traj_row: Row from data_trajectory.csv
        data_df: DataFrame from data.csv
        actor_dir: Path to actor directory
        planner: Planner name (e.g., 'dwa', 'teb', 'mppi', 'ddp')

    Returns:
        List of steps in APPLR pickle format
    """
    method = traj_row['Method']
    start_frame = int(traj_row['Start_frame_id'])
    done_frame = int(traj_row['Done_frame_id'])

    # Extract trajectory metadata
    collision = int(traj_row['Collision'])
    recovery = float(traj_row['Recovery'])
    smoothness = float(traj_row['Smoothness'])
    status = str(traj_row['Status'])
    total_time = float(traj_row['Time'])
    world_name = str(traj_row['World'])

    # Extract parameters for all frames
    params_list = extract_parameters_from_data_csv(data_df, start_frame, done_frame, method, planner)

    if len(params_list) == 0:
        print(f"Warning: No parameters found for trajectory {method}_{start_frame}_{done_frame}")
        return []

    # Build pickle trajectory
    pickle_traj = []
    num_steps = len(params_list)

    # Get delta_goal for reward calculation
    traj_data = data_df[
        (data_df['Method'] == method) &
        (data_df['img_label'] >= start_frame) &
        (data_df['img_label'] <= done_frame)
    ].sort_values('img_label')

    for step_idx, (params, (_, row)) in enumerate(zip(params_list, traj_data.iterrows())):
        frame_id = int(row['img_label'])

        # [0] Extract laser scan from image
        image_filename = f"{method}_{frame_id:06d}.png"
        image_path = os.path.join(actor_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}, skipping frame {frame_id}")
            continue

        laser_scan = extract_red_pixels_to_laserscan(image_path)

        # [1] Parameters (already extracted)
        # params is np.ndarray from params_list

        # [2] Reward: delta_goal * 10
        delta_goal = float(row['delta_goal_dist']) if 'delta_goal_dist' in row else 0.0
        reward = delta_goal * 10.0

        # [3] Done: True only for last step
        done = (step_idx == num_steps - 1)

        # [4] Info dict
        # Time accumulation: 0.5, 1.0, 1.5, 2.0, ...
        step_time = 0.5 + step_idx * 0.5

        info = {
            'world': world_name,
            'time': step_time,
            'collision': collision if done else 0,  # Only record collision in last step
            'status': status if done else 'running',
            'recovery': recovery if done else 0.0,
            'smoothness': smoothness if done else 0.0
        }

        # [5] Optimal time (placeholder, 0 for all)
        opt_time = 0.0

        # [6] Navigation metric (placeholder, 0 for all)
        nav_metric = 0.0

        # Create step
        step = [
            laser_scan,   # [0] (1, 721)
            params,       # [1] (num_params,)
            reward,       # [2] float
            done,         # [3] bool
            info,         # [4] dict
            opt_time,     # [5] float
            nav_metric    # [6] float
        ]

        pickle_traj.append(step)

    return pickle_traj


def process_all_trajectories(save_trajectory_csv: str, input_dir: str, output_dir: str, planner: str, actor_filter: List[str] = None):
    """
    Process all trajectories from save_trajectory.csv.

    Args:
        save_trajectory_csv: Path to save_trajectory.csv
        input_dir: Directory containing actor_* subdirectories
        output_dir: Output directory for pickle files
        planner: Planner name (e.g., 'dwa', 'teb', 'mppi', 'ddp')
        actor_filter: List of actor names to process (e.g., ['actor_0', 'actor_1'])
    """
    print(f"Loading {save_trajectory_csv}...")
    print(f"Planner: {planner.upper()}")

    # Read CSV with error handling for malformed lines
    try:
        traj_df = pd.read_csv(save_trajectory_csv, on_bad_lines='skip', engine='python')
        print(f"Found {len(traj_df)} total trajectories")
    except Exception as e:
        print(f"Error reading CSV with pandas, trying with error_bad_lines=False...")
        # Fallback for older pandas versions
        traj_df = pd.read_csv(save_trajectory_csv, error_bad_lines=False, warn_bad_lines=True)
        print(f"Found {len(traj_df)} total trajectories")

    # Filter by actor if specified
    if actor_filter:
        traj_df = traj_df[traj_df['actor_name'].isin(actor_filter)]
        print(f"Filtered to {len(traj_df)} trajectories for actors: {actor_filter}")

    # Group by actor
    actor_groups = traj_df.groupby('actor_name')

    print(f"\nProcessing {len(actor_groups)} actors")
    print()

    for actor_name, actor_traj_df in actor_groups:
        actor_dir = os.path.join(input_dir, actor_name)
        data_csv = os.path.join(actor_dir, 'data.csv')

        if not os.path.exists(actor_dir):
            print(f"Warning: {actor_dir} not found, skipping {actor_name}")
            continue

        if not os.path.exists(data_csv):
            print(f"Warning: {data_csv} not found, skipping {actor_name}")
            continue

        print(f"Processing {actor_name}...")
        print(f"  Found {len(actor_traj_df)} trajectories")

        # Load data.csv for this actor
        data_df = pd.read_csv(data_csv)

        # Create output directory for this actor
        actor_output_dir = os.path.join(output_dir, actor_name)
        os.makedirs(actor_output_dir, exist_ok=True)

        # Convert each trajectory
        for traj_idx, (_, row) in enumerate(actor_traj_df.iterrows(), 1):
            method = row['Method']
            start_frame = int(row['Start_frame_id'])
            done_frame = int(row['Done_frame_id'])

            print(f"  Trajectory {traj_idx}: {method}_{start_frame:06d} to {method}_{done_frame:06d}")

            pickle_traj = convert_trajectory_to_pickle(row, data_df, actor_dir, planner)

            if len(pickle_traj) > 0:
                output_path = os.path.join(actor_output_dir, f"traj_{traj_idx}.pickle")
                with open(output_path, 'wb') as f:
                    pickle.dump(pickle_traj, f)
                print(f"    Saved {len(pickle_traj)} steps to {output_path}")
            else:
                print(f"    Warning: Empty trajectory, skipped")

        print()


def main():
    parser = argparse.ArgumentParser(description='Convert CSV trajectories to APPLR pickle format')
    parser.add_argument('--input_dir', type=str,
                        default='/data/local/yl2832/appvlm/dwa_heurstic',
                        help='Directory containing actor_* subdirectories and save_trajectory.csv')
    parser.add_argument('--save_trajectory_csv', type=str,
                        default=None,
                        help='Path to save_trajectory.csv (default: <input_dir>/save_trajectory.csv)')
    parser.add_argument('--output_dir', type=str,
                        default='/data/local/yl2832/appvlm_ws/src/ros_jackal/buffer/dwa_heurstic_new/',
                        help='Output directory for pickle files')
    parser.add_argument('--planner', type=str,
                        default=None,
                        help='Planner name (dwa/teb/mppi/ddp). If not specified, infer from input_dir name')
    parser.add_argument('--actors', type=str, nargs='+',
                        help='Specific actors to process (e.g., actor_0 actor_1). If not specified, process all.')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine planner from input_dir if not specified
    if args.planner:
        planner = args.planner.lower()
    else:
        # Try to infer from directory name (e.g., dwa_heurstic -> dwa)
        input_dir_name = os.path.basename(args.input_dir.rstrip('/'))
        if 'dwa' in input_dir_name.lower():
            planner = 'dwa'
        elif 'teb' in input_dir_name.lower():
            planner = 'teb'
        elif 'mppi' in input_dir_name.lower():
            planner = 'mppi'
        elif 'ddp' in input_dir_name.lower():
            planner = 'ddp'
        else:
            print(f"Error: Cannot infer planner from directory name '{input_dir_name}'")
            print("Please specify --planner [dwa|teb|mppi|ddp]")
            return

    # Validate planner
    valid_planners = ['dwa', 'teb', 'mppi', 'ddp']
    if planner not in valid_planners:
        print(f"Error: Invalid planner '{planner}'. Must be one of: {valid_planners}")
        return

    # Determine save_trajectory.csv path
    if args.save_trajectory_csv:
        save_trajectory_csv = args.save_trajectory_csv
    else:
        save_trajectory_csv = os.path.join(args.input_dir, 'save_trajectory.csv')

    if not os.path.exists(save_trajectory_csv):
        print(f"Error: {save_trajectory_csv} not found!")
        return

    # Process all trajectories
    process_all_trajectories(save_trajectory_csv, args.input_dir, args.output_dir, planner, args.actors)

    print("Done!")


if __name__ == '__main__':
    main()
