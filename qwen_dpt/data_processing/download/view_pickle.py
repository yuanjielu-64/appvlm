#!/usr/bin/env python3
"""
Utility script to visualize and inspect pickle files in various formats.
"""

import pickle
import numpy as np
import argparse
import os
from typing import Any
import json


def load_pickle(pickle_path: str) -> Any:
    """Load pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def print_pickle_summary(data: Any, pickle_path: str):
    """Print a summary of the pickle file structure."""
    print("=" * 80)
    print(f"Pickle File: {pickle_path}")
    print("=" * 80)
    print(f"Type: {type(data).__name__}")

    if isinstance(data, list):
        print(f"Length: {len(data)} steps")
        print()

        if len(data) > 0:
            print("Step Structure:")
            step = data[0]
            if isinstance(step, list):
                print(f"  Each step is a list with {len(step)} elements:")
                for i, item in enumerate(step):
                    if isinstance(item, np.ndarray):
                        print(f"    [{i}] ndarray: shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, dict):
                        print(f"    [{i}] dict: keys={list(item.keys())}")
                    else:
                        print(f"    [{i}] {type(item).__name__}: {item}")
            else:
                print(f"  Each step is: {type(step).__name__}")

    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")

    print()


def print_pickle_detailed(data: Any, num_steps: int = 3):
    """Print detailed information of first/middle/last steps."""
    if not isinstance(data, list):
        print("Data is not a trajectory (list of steps)")
        return

    print("=" * 80)
    print("Detailed Step Information")
    print("=" * 80)

    # Determine which steps to show
    indices = []
    if len(data) > 0:
        indices.append(0)  # First
    if len(data) > 2:
        indices.append(len(data) // 2)  # Middle
    if len(data) > 1:
        indices.append(len(data) - 1)  # Last

    for idx in indices[:num_steps]:
        step = data[idx]
        print(f"\n{'─' * 80}")
        print(f"Step {idx} / {len(data) - 1}")
        print(f"{'─' * 80}")

        if not isinstance(step, list):
            print(f"  Type: {type(step).__name__}")
            continue

        for i, item in enumerate(step):
            if isinstance(item, np.ndarray):
                print(f"  [{i}] ndarray:")
                print(f"      Shape: {item.shape}")
                print(f"      Dtype: {item.dtype}")
                print(f"      Range: [{item.min():.6f}, {item.max():.6f}]")
                print(f"      Mean: {item.mean():.6f}")
                if item.size <= 20:
                    print(f"      Values: {item}")
                else:
                    print(f"      First 10: {item.flatten()[:10]}")

            elif isinstance(item, dict):
                print(f"  [{i}] dict:")
                for key, value in item.items():
                    if isinstance(value, float):
                        print(f"      {key}: {value:.6f}")
                    else:
                        print(f"      {key}: {value}")

            elif isinstance(item, bool):
                print(f"  [{i}] bool: {item}")

            elif isinstance(item, (int, float)):
                print(f"  [{i}] {type(item).__name__}: {item}")

            else:
                print(f"  [{i}] {type(item).__name__}: {item}")

        print()


def print_laser_scan_stats(data: Any):
    """Print statistics about laser scans in the trajectory."""
    if not isinstance(data, list) or len(data) == 0:
        return

    print("=" * 80)
    print("Laser Scan Statistics")
    print("=" * 80)

    laser_scans = []
    for step in data:
        if isinstance(step, list) and len(step) > 0:
            laser = step[0]
            if isinstance(laser, np.ndarray):
                laser_scans.append(laser)

    if len(laser_scans) == 0:
        print("No laser scans found")
        return

    laser_scans = np.array(laser_scans)
    print(f"Total scans: {len(laser_scans)}")
    print(f"Scan shape: {laser_scans.shape}")
    print(f"Overall range: [{laser_scans.min():.6f}, {laser_scans.max():.6f}]")
    print(f"Overall mean: {laser_scans.mean():.6f}")

    # Count beams at max range (free space)
    max_range = 0.5
    free_beams = np.sum(laser_scans >= max_range)
    total_beams = laser_scans.size
    print(f"Free space beams (>= {max_range}): {free_beams} / {total_beams} ({100*free_beams/total_beams:.1f}%)")

    # Closest obstacle per step
    print("\nClosest obstacle per step:")
    for i, scan in enumerate(laser_scans):
        min_dist = scan.min()
        print(f"  Step {i}: {min_dist:.6f}")

    print()


def export_to_json(data: Any, output_path: str):
    """Export pickle data to JSON format (for visualization)."""

    def convert_to_serializable(obj):
        """Convert numpy arrays and other non-serializable types to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj

    json_data = convert_to_serializable(data)

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Exported to JSON: {output_path}")


def visualize_trajectory_info(data: Any):
    """Print trajectory-level information."""
    if not isinstance(data, list) or len(data) == 0:
        return

    print("=" * 80)
    print("Trajectory Information")
    print("=" * 80)

    # Extract info from last step
    last_step = data[-1]
    if isinstance(last_step, list) and len(last_step) > 4:
        info = last_step[4]
        if isinstance(info, dict):
            print("Final Status:")
            print(f"  Status: {info.get('status', 'N/A')}")
            print(f"  Time: {info.get('time', 'N/A')}")
            print(f"  Collision: {info.get('collision', 'N/A')}")
            print(f"  World: {info.get('world', 'N/A')}")

            if 'recovery' in info:
                print(f"  Recovery: {info['recovery']:.6f}")
            if 'smoothness' in info:
                print(f"  Smoothness: {info['smoothness']:.6f}")

    # Compute total reward
    total_reward = 0.0
    for step in data:
        if isinstance(step, list) and len(step) > 2:
            reward = step[2]
            if isinstance(reward, (int, float)):
                total_reward += reward

    print(f"\nTotal Reward: {total_reward:.3f}")
    print(f"Trajectory Length: {len(data)} steps")

    # Check if trajectory completed successfully
    if isinstance(last_step, list) and len(last_step) > 3:
        done = last_step[3]
        print(f"Completed: {done}")

    print()


def main():
    parser = argparse.ArgumentParser(description='View and analyze pickle files')
    parser.add_argument('--pickle_path', type=str, default= "/home/yuanjielu/robot_navigation/noetic/applr/src/ros_jackal/buffer/dwa_cluster/actor_0/traj_1.pickle", help='Path to pickle file')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument('--detailed', action='store_true', help='Show detailed step information')
    parser.add_argument('--laser', action='store_true', help='Show laser scan statistics')
    parser.add_argument('--trajectory', action='store_true', help='Show trajectory information')
    parser.add_argument('--export-json', type=str, help='Export to JSON file')
    parser.add_argument('--steps', type=int, default=3, help='Number of steps to show in detail (default: 3)')
    parser.add_argument('--all', action='store_true', help='Show all information')

    args = parser.parse_args()

    if not os.path.exists(args.pickle_path):
        print(f"Error: File not found: {args.pickle_path}")
        return

    # Load pickle
    data = load_pickle(args.pickle_path)

    # Default: show everything if no specific option selected
    show_all = args.all or not (args.summary or args.detailed or args.laser or args.trajectory or args.export_json)

    # Show requested information
    if args.summary or show_all:
        print_pickle_summary(data, args.pickle_path)

    if args.trajectory or show_all:
        visualize_trajectory_info(data)

    if args.detailed or show_all:
        print_pickle_detailed(data, args.steps)

    if args.laser or show_all:
        print_laser_scan_stats(data)

    if args.export_json:
        export_to_json(data, args.export_json)


if __name__ == '__main__':
    main()
