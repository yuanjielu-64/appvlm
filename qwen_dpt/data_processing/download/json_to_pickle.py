#!/usr/bin/env python3
"""
Convert JSON trajectory data with costmap images to pickle format compatible with APPLR buffer.

This script reads actor_*.json files, extracts trajectories, converts red pixels in costmap images
to laser scans, and saves them as pickle files in APPLR format.
"""

import json
import pickle
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def extract_red_pixels_to_laserscan(image_path: str, num_beams: int = 721, max_range: float = 0.5) -> np.ndarray:
    """
    Extract red pixels from costmap image and convert to laser scan format.

    This simulates a 270° FOV laser scanner by raycasting from the robot center.
    Mimics baseline APPLR format: (1, 721) array with normalized distances.

    Args:
        image_path: Path to the costmap image
        num_beams: Number of laser scan beams (default 721 for 270° FOV)
        max_range: Maximum laser range in normalized units (default 0.5)

    Returns:
        laser_scan: Array of shape (1, 721) with normalized distances
    """
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    pixels = np.array(img)

    # Extract red channel and identify red pixels (obstacles)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]

    # Red pixels: R > 200 and G < 100 and B < 100
    red_mask = (red > 200) & (green < 100) & (blue < 100)

    # Find robot center (yellow square)
    yellow_mask = (red > 200) & (green > 200) & (blue < 100)
    if np.any(yellow_mask):
        robot_y, robot_x = np.where(yellow_mask)
        center_x = int(np.mean(robot_x))
        center_y = int(np.mean(robot_y))
    else:
        # Fallback to image center
        center_x = width // 2
        center_y = height // 2

    # Initialize laser scan with max range
    laser_scan = np.full(num_beams, max_range, dtype=np.float64)

    # 270° FOV: from -135° to +135° relative to robot's forward direction
    # Forward (green arrow) is +x in image space
    fov_rad = np.deg2rad(270)
    start_angle = -fov_rad / 2  # -135°
    end_angle = fov_rad / 2     # +135°

    # For each beam, raycast to find nearest obstacle
    for beam_idx in range(num_beams):
        # Calculate beam angle relative to robot forward (+x direction)
        angle = start_angle + (beam_idx / (num_beams - 1)) * fov_rad

        # Raycast from robot center along this angle
        # x-axis = forward, y-axis = left (image y is down, so negate)
        dx_unit = np.cos(angle)
        dy_unit = -np.sin(angle)  # Negate because image y-axis is down

        # Search along ray up to max distance
        max_pixel_dist = int(np.sqrt(width**2 + height**2))

        for dist_pixels in range(1, max_pixel_dist):
            x = int(center_x + dx_unit * dist_pixels)
            y = int(center_y + dy_unit * dist_pixels)

            # Check bounds
            if x < 0 or x >= width or y < 0 or y >= height:
                break

            # Check if hit obstacle (red pixel)
            if red_mask[y, x]:
                # Normalize distance
                # Assuming costmap represents ~10m x 10m area (typical for local planning)
                # and max_range=0.5 corresponds to ~30m in real world
                # So pixel distance needs to be normalized appropriately
                pixels_per_meter = width / 10.0  # Assume 10m x 10m costmap
                dist_meters = dist_pixels / pixels_per_meter
                dist_normalized = min(dist_meters / 60.0, max_range)  # 60m = 1.0 normalized

                laser_scan[beam_idx] = dist_normalized
                break

    return laser_scan.reshape(1, num_beams)


def parse_trajectory_id(entry_id: str) -> Tuple[str, int]:
    """Parse entry id like 'HB_003741' into prefix and frame number."""
    parts = entry_id.split('_')
    return parts[0], int(parts[1])


def find_trajectories(json_data: List[Dict]) -> List[List[Dict]]:
    """
    Group JSON entries into continuous trajectories based on sequential frame numbers.

    Args:
        json_data: List of JSON entry dictionaries

    Returns:
        List of trajectories, where each trajectory is a list of entries
    """
    trajectories = []
    current_traj = []
    prev_prefix = None
    prev_num = None

    for entry in json_data:
        prefix, num = parse_trajectory_id(entry['id'])

        # Check if this continues the current trajectory
        if prev_prefix is None or (prefix == prev_prefix and num == prev_num + 1):
            current_traj.append(entry)
        else:
            # Start new trajectory
            if len(current_traj) > 0:
                trajectories.append(current_traj)
            current_traj = [entry]

        prev_prefix = prefix
        prev_num = num

    # Add last trajectory
    if len(current_traj) > 0:
        trajectories.append(current_traj)

    return trajectories


def convert_trajectory_to_pickle(trajectory: List[Dict], image_base_dir: str) -> List:
    """
    Convert a trajectory from JSON format to APPLR pickle format.

    APPLR pickle format (list of steps):
    Each step is a list: [obs, act, reward, done, info, opt_time, nav_metric]
    - obs: observation (laser_scan or state, depending on format)
    - act: action (parameters)
    - reward: float
    - done: bool
    - info: dict with keys ['world', 'time', 'collision', 'status', 'recovery', 'smoothness']
    - opt_time: optimal time (only meaningful in last step)
    - nav_metric: navigation metric (only meaningful in last step)

    Note: JSON data is missing many fields that baseline APPLR has:
    ✅ Available: costmap images, parameters, linear/angular velocity
    ❌ Missing: laser scan (extracted from image), reward, collision, success status, goal position

    Args:
        trajectory: List of JSON entries forming one trajectory
        image_base_dir: Directory containing the images

    Returns:
        List of steps in APPLR pickle format
    """
    pickle_traj = []

    for step_idx, entry in enumerate(trajectory):
        # ========== [0] Observation ==========
        # Extract laser scan from costmap image (red pixels -> raycasting)
        image_rel_path = entry['images'][0]
        image_path = os.path.join(image_base_dir, os.path.basename(image_rel_path))

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        laser_scan = extract_red_pixels_to_laserscan(image_path)  # (1, 721)

        # ========== [1] Action ==========
        # DWA parameters from JSON
        params = np.array(entry['parameters'], dtype=np.float32)

        # ========== [2] Reward ==========
        # ❌ NOT AVAILABLE in JSON - use placeholder
        reward = 0.0

        # ========== [3] Done ==========
        # Assume last step in trajectory is done
        done = (step_idx == len(trajectory) - 1)

        # ========== [4] Info Dict ==========
        # Parse velocity from conversations (for completeness)
        conv_text = entry['conversations'][0]
        import re
        lin_match = re.search(r'Linear velocity:\s*([\d.]+)', conv_text)
        ang_match = re.search(r'Angular velocity:\s*([-\d.]+)', conv_text)

        linear_vel = float(lin_match.group(1)) if lin_match else 0.0
        angular_vel = float(ang_match.group(1)) if ang_match else 0.0

        info = {
            'world': 'unknown.world',  # ❌ NOT AVAILABLE
            'time': step_idx * 0.5,    # ❌ ESTIMATED (assume 2Hz control)
            'collision': 0,            # ❌ NOT AVAILABLE (assume no collision)
            'status': 'success' if done else 'running',  # ❌ GUESSED
            'recovery': 0.0,           # ❌ NOT AVAILABLE
            'smoothness': 0.0,         # ❌ NOT AVAILABLE
            # Extra fields for reference
            'linear_vel': linear_vel,
            'angular_vel': angular_vel
        }

        # ========== [5] Optimal Time ==========
        # Only meaningful in last step (baseline uses this for scoring)
        opt_time = 0.0  # ❌ NOT AVAILABLE

        # ========== [6] Navigation Metric ==========
        # Only meaningful in last step (baseline uses this for scoring)
        nav_metric = 0.0  # ❌ NOT AVAILABLE

        # Create step
        step = [
            laser_scan,   # [0] (1, 721) - ✅ EXTRACTED from image
            params,       # [1] (8,) for DWA - ✅ FROM JSON
            reward,       # [2] float - ❌ PLACEHOLDER
            done,         # [3] bool - ✅ INFERRED
            info,         # [4] dict - ⚠️ PARTIAL (missing many fields)
            opt_time,     # [5] float - ❌ PLACEHOLDER
            nav_metric    # [6] float - ❌ PLACEHOLDER
        ]

        pickle_traj.append(step)

    return pickle_traj


def process_actor(actor_dir: str, output_dir: str):
    """
    Process all trajectories from one actor's JSON file.

    Args:
        actor_dir: Path to actor directory (e.g., actor_0)
        output_dir: Output directory for pickle files
    """
    actor_name = os.path.basename(actor_dir)
    json_path = os.path.join(actor_dir, f"{actor_name}.json")

    if not os.path.exists(json_path):
        print(f"Warning: JSON file not found: {json_path}")
        return

    print(f"Processing {actor_name}...")

    # Load JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Find trajectories
    trajectories = find_trajectories(json_data)
    print(f"  Found {len(trajectories)} trajectories")

    # Create output directory for this actor
    actor_output_dir = os.path.join(output_dir, actor_name)
    os.makedirs(actor_output_dir, exist_ok=True)

    # Convert each trajectory
    for traj_idx, trajectory in enumerate(trajectories):
        print(f"  Converting trajectory {traj_idx + 1} ({len(trajectory)} steps)...")

        pickle_traj = convert_trajectory_to_pickle(trajectory, actor_dir)

        if len(pickle_traj) > 0:
            output_path = os.path.join(actor_output_dir, f"traj_{traj_idx + 1}.pickle")
            with open(output_path, 'wb') as f:
                pickle.dump(pickle_traj, f)
            print(f"    Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert JSON trajectories to APPLR pickle format')
    parser.add_argument('--input_dir', type=str,
                        default='/home/yuanjielu/robot_navigation/noetic/app_data/ddp_heurstic',
                        help='Directory containing actor_* subdirectories')
    parser.add_argument('--output_dir', type=str,
                        default='./test',
                        help='Output directory for pickle files')
    parser.add_argument('--actors', type=str, nargs='+',
                        help='Specific actors to process (e.g., actor_0 actor_1). If not specified, process all.')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find actor directories
    if args.actors:
        actor_dirs = [os.path.join(args.input_dir, actor) for actor in args.actors]
    else:
        actor_dirs = sorted([
            os.path.join(args.input_dir, d)
            for d in os.listdir(args.input_dir)
            if d.startswith('actor_') and os.path.isdir(os.path.join(args.input_dir, d))
        ])

    print(f"Found {len(actor_dirs)} actors to process")

    # Process each actor
    for actor_dir in actor_dirs:
        process_actor(actor_dir, args.output_dir)

    print("Done!")


if __name__ == '__main__':
    main()
