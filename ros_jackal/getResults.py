#!/usr/bin/env python3
"""
BARN Challenge Environment Analysis Tool

This script analyzes baseline navigation results to identify the most challenging 
environments based on completion time and success rates.
"""

import os
import pandas as pd
import numpy as np
from os.path import join, exists
import random

def path_coord_to_gazebo_coord(x, y):
    """Convert path coordinates to Gazebo coordinates"""
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5

    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift

    return (gazebo_x, gazebo_y)


def compute_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_navigation_metric(world_id, actual_time, success, base_path="src/barn_challenge_lu"):
    """
    Calculate navigation metric for a given world and run
    
    Args:
        world_id: World identifier (0-299: static, 300-359: dynamic)
        actual_time: Actual completion time in seconds
        success: Whether the navigation was successful (1/0)
        base_path: Base path to BARN challenge code
        
    Returns:
        nav_metric: Navigation score
        optimal_time: Optimal completion time
        path_length: Path length in meters
    """
    # Set environment-specific coordinate systems
    if world_id < 300:  # Static environments (0-299)
        INIT_POSITION = [-2.25, 3, 1.57]
        GOAL_POSITION = [0, GOAL]
    elif world_id < 360:  # Dynamic environments (300-359)
        INIT_POSITION = [11, 0, 3.14]
        GOAL_POSITION = [GOAL, 0]
    else:
        raise ValueError(f"World index {world_id} does not exist")

    if world_id >= 300:  # DynaBARN environment without planned path
        path_length = abs(GOAL_POSITION[0] - INIT_POSITION[0])
    else:
        path_file_name = join("jackal_helper/worlds/BARN1/path_files", f"path_{world_id}.npy")

        if not exists(path_file_name):
            # Use default distance if path file doesn't exist
            path_length = GOAL_POSITION[0] - INIT_POSITION[0]
        else:
            try:
                path_array = np.load(path_file_name)
                path_array = [path_coord_to_gazebo_coord(*p) for p in path_array]
                path_array = np.insert(path_array, 0, (INIT_POSITION[0], INIT_POSITION[1]), axis=0)
                path_array = np.insert(path_array, len(path_array),
                                     (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1]), axis=0)
                path_length = 0
                for p1, p2 in zip(path_array[:-1], path_array[1:]):
                    path_length += compute_distance(p1, p2)
            except Exception as e:
                print(f"Warning: Unable to read path file {path_file_name}: {e}")
                path_length = GOAL_POSITION[0] - INIT_POSITION[0]

    # Navigation metric: success * optimal_time / clip(actual_time, 2 * optimal_time, 8 * optimal_time)
    optimal_time = path_length / 2
    nav_metric = int(success) * optimal_time / np.clip(actual_time, 2 * optimal_time, 8 * optimal_time)

    return nav_metric, optimal_time, path_length


def collect_test_log_data(log_file_path, top_n_per_env=20):
    """
    Collect data from test log files
    
    Args:
        log_file_path: Path to the test log file
        top_n_per_env: Number of top results per environment to analyze

    Returns:
        combined_df: Combined data DataFrame
        processed_files: File mapping {world_id: file_path}
        missing_files: List of missing file IDs
    """
    print(f"Reading log file: {log_file_path}")
    print(f"Taking top {top_n_per_env} results per environment")
    print("=" * 60)

    all_data = []
    world_data = {}  # {world_id: [records]}

    try:
        with open(log_file_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 7:  # world_idx success collided timeout actual_time optimal_time nav_metric
                    world_id = int(parts[0])
                    success = int(parts[1])
                    collided = int(parts[2])
                    timeout = int(parts[3])
                    actual_time = float(parts[4])
                    optimal_time = float(parts[5])
                    nav_metric = float(parts[6])

                    # Determine status
                    if success:
                        status = 'success'
                    elif collided:
                        status = 'collision'
                    elif timeout:
                        status = 'timeout'
                    else:
                        status = 'unknown'

                    record = {
                        'world_id': world_id,
                        'Time': actual_time,
                        'Status': status,
                        'nav_metric': nav_metric,
                        'optimal_time': optimal_time
                    }

                    if world_id not in world_data:
                        world_data[world_id] = []
                    world_data[world_id].append(record)

    except Exception as e:
        print(f"❌ Error: Failed to read log file: {e}")
        return None, {}, []

    # Take top N records per environment
    processed_files = {}
    for world_id, records in world_data.items():
        # Only take the first top_n_per_env records
        selected_records = records[:top_n_per_env]
        all_data.extend(selected_records)
        processed_files[world_id] = log_file_path

    if not all_data:
        print("❌ No valid data found")
        return None, {}, []

    # Convert to DataFrame
    combined_df = pd.DataFrame(all_data)

    # Check for missing world_ids (0-299)
    existing_worlds = set(world_data.keys())
    all_worlds = set(range(300))
    missing_files = list(all_worlds - existing_worlds)

    print(f"✅ Successfully collected data from {len(world_data)} environments, {len(combined_df)} total records")
    print(f"⚠️  Missing environments: {len(missing_files)}")
    if missing_files:
        print("Missing environment details:")
        for world_id in missing_files:
            print(f"  Environment {world_id}: Missing in log file")

    return combined_df, processed_files, missing_files


def collect_baseline_data(baseline_dir, top_n_per_env=20, time_threshold=50):
    """
    Collect all baseline data

    Args:
        baseline_dir: Directory containing baseline CSV files
        top_n_per_env: Number of top results per environment to analyze
        time_threshold: Time threshold for marking as missing (default: 30 seconds)

    Returns:
        combined_df: Combined data DataFrame
        processed_files: File mapping {world_id: file_path}
        missing_files: List of missing file IDs
        time_missing_files: List of files with time > threshold
    """
    print(f"📂 Collecting from directory: {baseline_dir}")
    print(f"📊 Taking top {top_n_per_env} results per environment")
    print(f"⏰ Time threshold for missing: {time_threshold} seconds")
    print("=" * 60)

    target_envs = [279, 256, 138, 16, 280, 2, 275, 252, 227, 181, 185, 282, 287, 234, 277, 289,
                   59, 206, 112, 250, 49, 221, 216, 121, 266, 186, 143, 123, 295, 245, 85, 240,
                   211, 271, 241, 209, 204, 254, 78, 119, 286, 212, 180, 30, 192, 118, 69, 132,
                   298, 172, 244, 217, 293, 175, 210, 58, 162, 260, 171, 157, 86, 230, 140, 229,
                   265, 231, 163, 215, 294, 197]

    all_data = []
    missing_files = []
    time_missing_files = []
    processed_files = {}

    # Iterate through all files 0-299
    for world_id in range(300):

        if world_id in target_envs:
            continue

        # Try multiple filename formats
        possible_files = [
            f"baseline_results_{world_id}.csv",
            f"test_results_{world_id}.csv",
            f"results_{world_id}.csv",
            f"{world_id}.csv"
        ]

        csv_file = None
        for filename in possible_files:
            test_path = join(baseline_dir, filename)
            if exists(test_path):
                csv_file = test_path
                break

        if csv_file is None:
            missing_files.append(world_id)
            continue

        try:
            # Read CSV file
            df = pd.read_csv(csv_file)

            if df.empty:
                print(f"⚠️  Warning: {csv_file} is empty")
                continue

            # Only take the first top_n_per_env rows of data
            df = df.head(top_n_per_env)

            # Check if required columns exist
            if 'Time' not in df.columns or 'Status' not in df.columns:
                missing_cols = []
                if 'Time' not in df.columns:
                    missing_cols.append('Time')
                if 'Status' not in df.columns:
                    missing_cols.append('Status')
                print(f"Warning: Missing columns in {csv_file}: {missing_cols}")
                continue

            # 新增逻辑：检查是否有时间超过阈值的记录
            if 'Time' in df.columns:
                avg_time = df['Time'].mean()
                max_time = df['Time'].max()

                # 如果平均时间或最大时间超过阈值，标记为time missing
                if avg_time > time_threshold or max_time > time_threshold:
                    time_missing_files.append({
                        'world_id': world_id,
                        'avg_time': avg_time,
                        'max_time': max_time,
                        'file_path': csv_file
                    })
                    print(f"⏰ Time missing: World {world_id}, avg_time: {avg_time:.2f}s, max_time: {max_time:.2f}s")

                    # 可以选择是否跳过这些环境
                    # continue  # 如果要跳过超时环境，取消注释这行

            # Add world_id column and collect raw data
            df['world_id'] = world_id
            all_data.append(df[['world_id', 'Time', 'Status']].copy())
            processed_files[world_id] = csv_file

        except Exception as e:
            print(f"Error: Failed to read {csv_file}: {e}")

    if not all_data:
        print("No valid data files found")
        return None, {}, missing_files, time_missing_files

    # Merge all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Successfully collected data from {len(all_data)} files, {len(combined_df)} total records")
    print(f"Number of missing files: {len(missing_files)}")
    print(f"Number of time-missing files: {len(time_missing_files)}")

    if missing_files:
        print("Missing baseline file details:")
        for world_id in missing_files:
            print(f"  World {world_id}: No CSV file found in {baseline_dir}")

    if time_missing_files:
        print("Time-missing baseline file details:")
        for item in time_missing_files:
            print(f"  World {item['world_id']}: avg_time={item['avg_time']:.2f}s, max_time={item['max_time']:.2f}s")

    return combined_df, processed_files, missing_files, time_missing_files


def calculate_environment_statistics(combined_df, processed_files, use_existing_nav_metric=False):

    print("📊 Computing navigation statistics...")

    def calculate_nav_metric_vectorized(row):
        """Vectorized function to calculate navigation metrics for each row"""
        # success = 1 if row['Status'] == 'success' else 0

        success = 1 if (row['Status'] == 'success' ) else 0

        nav_metric, optimal_time, path_length = calculate_navigation_metric(
            row['world_id'], row['Time'], success
        )

        return pd.Series({
            'nav_metric': nav_metric,
            'optimal_time': optimal_time,
            'path_length': path_length,
            'success': success
        })

    # Apply vectorized computation
    nav_data = combined_df.apply(calculate_nav_metric_vectorized, axis=1)
    combined_df = pd.concat([combined_df, nav_data], axis=1)

    # Save raw results in BARN challenge format
    suffix = f"_{GOAL}"
    output_file = f"results/results{suffix}.txt"
    
    print(f"💾 Saving raw results to: {output_file}")
    with open(output_file, "w") as f:
        # Write header comment
        f.write("# Format: world_id success collided timeout actual_time optimal_time nav_metric\n")
        
        for _, row in combined_df.iterrows():
            timeout = 1 if row['Time'] >= 100 else 0
            collided = 1 if row['Status'] == 'collision' else 0
            f.write(f"{int(row['world_id'])} {int(row['success'])} {collided} {timeout} "
                   f"{row['Time']:.4f} {row['optimal_time']:.4f} {row['nav_metric']:.4f}\n")

    # Calculate statistics grouped by world_id
    results = []

    for world_id, group in combined_df.groupby('world_id'):
        # Basic statistics
        total_episodes = len(group)

        processed_times = []
        for _, row in group.iterrows():
            if row['success'] == 1:
                processed_times.append(row['Time'])
            else:
                processed_times.append(50.0)

        if total_episodes > 10:  # Only apply trimming if we have enough data
            sorted_processed_times = sorted(processed_times)
            trimmed_processed_times = sorted_processed_times[5:-5]
            avg_time = np.mean(trimmed_processed_times)
        else:
            avg_time = np.mean(processed_times)

        # avg_time = group['Time'].mean()
        avg_nav_metric = group['nav_metric'].mean()
        avg_optimal_time = group['optimal_time'].mean()
        avg_path_length = group['path_length'].mean()

        # success_group = group[group['Status'] == 'success']
        success_group = group[group['success'] == 1]
        if not success_group.empty and len(success_group) > 10:
            sorted_success_times = success_group['Time'].sort_values()
            trimmed_success_times = sorted_success_times.iloc[5:-5]
            avg_success_time = trimmed_success_times.mean() if not trimmed_success_times.empty else None
        else:
            avg_success_time = success_group['Time'].mean() if not success_group.empty else None

        # avg_success_time = success_group['Time'].mean() if not success_group.empty else None

        # Status statistics
        status_counts = group['Status'].value_counts()
        status_stats = {}
        for status, count in status_counts.items():
            status_stats[status] = {
                'count': int(count),
                'percentage': (count / total_episodes) * 100
            }

        # Success rate
        success_rate = status_stats.get('success', {}).get('percentage', 0)

        results.append({
            'world_id': int(world_id),
            'avg_time': avg_time,
            'avg_success_time': avg_success_time,
            'avg_nav_metric': avg_nav_metric,
            'avg_optimal_time': avg_optimal_time,
            'avg_path_length': avg_path_length,
            'num_episodes': total_episodes,
            'success_rate': success_rate,
            'status_stats': status_stats,
            'csv_file': processed_files[world_id]
        })

    print(f"Calculation complete, processed {len(results)} environments")
    return results


def analyze_baseline_results(baseline_dir, top_n_per_env):
    # Step 1: Collect data
    combined_df, processed_files, missing_files, time_missing_files = collect_baseline_data(baseline_dir, top_n_per_env)

    if combined_df is None:
        return []

    # Step 2: Calculate statistics
    results = calculate_environment_statistics(combined_df, processed_files)

    # Step 3: Sort
    sorted_results = sorted(results, key=lambda x: x['avg_time'], reverse=True)

    # Print statistics
    print(f"\nStatistics:")
    print(f"Successfully processed files: {len(results)}")
    print(f"Missing files: {len(missing_files)}")
    if missing_files:
        print(f"Missing file IDs: {missing_files}")
        print("Missing file details:")
        for world_id in missing_files:
            print(f"  World {world_id}: No baseline file found")

    # Print sorted results
    print(f"\nEnvironment difficulty ranking (sorted by average time from high to low):")
    print("-" * 120)
    print(f"{'Rank':<4} {'World ID':<8} {'Avg Time':<10} {'Nav Score':<10} {'Success Rate':<12} {'Test Count':<10} {'Status Distribution':<50}")
    print("-" * 120)

    for rank, result in enumerate(sorted_results[:20], 1):  # Show top 20 most difficult
        # Build status distribution string
        status_str = ""
        for status, stats in result['status_stats'].items():
            status_str += f"{status}:{stats['count']}({stats['percentage']:.1f}%) "

        print(
            f"{rank:<4} {result['world_id']:<8} {result['avg_time']:<9.3f} {result['avg_nav_metric']:<9.3f} {result['success_rate']:<11.1f}% {result['num_episodes']:<10} {status_str:<50}")

    if len(sorted_results) > 20:
        print(f"... (total {len(sorted_results)} environments)")

    for rank, result in enumerate(reversed(sorted_results[-100:]), 1):
        status_str = ""
        for status, stats in result['status_stats'].items():
            status_str += f"{status}:{stats['count']}({stats['percentage']:.1f}%) "

        print(
            f"{rank:<4} {result['world_id']:<8} {result['avg_time']:<9.3f} {result['avg_nav_metric']:<9.3f} {result['success_rate']:<11.1f}% {result['num_episodes']:<10} {status_str:<50}")

    return sorted_results

def save_results(sorted_results, output_file="hard_environments.csv"):
    """
    Save results to CSV file

    Args:
        sorted_results: Sorted results
        output_file: Output filename
    """

    if not sorted_results:
        print("No results to save")
        return

    # Prepare data to save
    save_data = []
    for i, result in enumerate(sorted_results, 1):
        row = {
            'rank': i,
            'world_id': result['world_id'],
            'avg_time': result['avg_time'],
            'avg_nav_metric': result['avg_nav_metric'],
            'avg_optimal_time': result['avg_optimal_time'],
            'avg_path_length': result['avg_path_length'],
            'success_rate': result['success_rate'],
            'num_episodes': result['num_episodes']
        }

        # Add statistics for various statuses
        for status, stats in result['status_stats'].items():
            row[f'{status}_count'] = stats['count']
            row[f'{status}_percentage'] = stats['percentage']

        save_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(save_data)

    # Save to file
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


def get_hardest_environments(sorted_results, top_n=50):
    # 从前100个最难的环境中随机选50个
    hard_pool = sorted_results[:150]
    selected_hard = random.sample(hard_pool, 80)
    hard_ids = [result['world_id'] for result in selected_hard]

    # 从后200个环境中随机选100个
    easy_pool = sorted_results[150:300]
    selected_easy = random.sample(easy_pool, 20)
    easy_ids = [result['world_id'] for result in selected_easy]

    # 合并
    all_ids = hard_ids + hard_ids

    print(f"\nSelected 50 hard environments from top 100: {hard_ids}")
    print(f"Selected 100 easy environments from remaining 200: {easy_ids}")
    print(f"Total training environments: {len(all_ids)}")

    return all_ids


def analyze_status_distribution(sorted_results, top_n_per_env=20):

    print("\n" + "=" * 80)
    print("Overall Status Distribution Statistics:")
    print("=" * 80)

    all_status_counts = {}
    total_episodes = 0

    for result in sorted_results:
        total_episodes += result['num_episodes']
        for status, stats in result['status_stats'].items():
            if status not in all_status_counts:
                all_status_counts[status] = 0
            all_status_counts[status] += stats['count']

    # Add overall average time statistics
    print(f"Overall statistics (based on top {top_n_per_env} results per environment):")
    print("-" * 40)

    # Calculate average time and navigation scores for all environments
    all_avg_times = [r['avg_time'] for r in sorted_results]
    all_nav_metrics = [r['avg_nav_metric'] for r in sorted_results]

    all_avg_times_cleaned = [0 if x is None else x for x in all_avg_times]

    overall_avg_time = np.mean(all_avg_times_cleaned)
    median_time = np.median(all_avg_times_cleaned)
    std_time = np.std(all_avg_times_cleaned)

    overall_avg_nav_metric = np.mean(all_nav_metrics)
    median_nav_metric = np.median(all_nav_metrics)
    std_nav_metric = np.std(all_nav_metrics)

    print(f"Overall average time for all environments: {overall_avg_time:.3f} seconds")
    print(f"Median time: {median_time:.3f} seconds")
    print(f"Time standard deviation: {std_time:.3f} seconds")
    print(f"Longest average time: {max(all_avg_times_cleaned):.3f} seconds (World {sorted_results[0]['world_id']})")
    print(f"Shortest average time: {min(all_avg_times_cleaned):.3f} seconds (World {sorted_results[-1]['world_id']})")

    print(f"\nNavigation score statistics:")
    print(f"Overall average navigation score for all environments: {overall_avg_nav_metric:.3f}")
    print(f"Navigation score median: {median_nav_metric:.3f}")
    print(f"Navigation score standard deviation: {std_nav_metric:.3f}")

    # Sort by navigation score to find highest and lowest
    sorted_by_nav = sorted(sorted_results, key=lambda x: x['avg_nav_metric'], reverse=True)
    print(f"Highest average navigation score: {sorted_by_nav[0]['avg_nav_metric']:.3f} (World {sorted_by_nav[0]['world_id']})")
    print(f"Lowest average navigation score: {sorted_by_nav[-1]['avg_nav_metric']:.3f} (World {sorted_by_nav[-1]['world_id']})")

    # Print overall distribution
    print(f"\nTotal test episodes: {total_episodes}")
    print("-" * 40)
    for status, count in sorted(all_status_counts.items()):
        percentage = (count / total_episodes) * 100
        print(f"{status:<15}: {count:>6} times ({percentage:>5.2f}%)")

    # 按成功率分析环境分布
    success_ranges = [(90, 100), (70, 90), (50, 70), (30, 50), (0, 30)]

    print(f"\nEnvironment distribution analysis by success rate:")
    print("-" * 40)
    for min_rate, max_rate in success_ranges:
        count = len([r for r in sorted_results
                     if min_rate <= r['success_rate'] < max_rate])
        print(f"{min_rate}%-{max_rate}% success rate: {count:>3} environments")

    # Analyze hardest environments (success rate < 50%)
    hard_envs = [r for r in sorted_results if r['success_rate'] < 50]
    if hard_envs:
        print(f"\nHigh difficulty environments (success rate < 50%) analysis:")
        print("-" * 60)
        print(f"Number of high difficulty environments: {len(hard_envs)}")
        print(f"Average success rate: {np.mean([r['success_rate'] for r in hard_envs]):.1f}%")
        print(f"Average completion time: {np.mean([r['avg_time'] for r in hard_envs]):.3f} seconds")

        # Show details of top 10 hardest environments
        print(f"\nTop 10 hardest environments details:")
        print("-" * 80)
        print(f"{'World ID':<8} {'Success Rate':<12} {'Avg Time':<10} {'Main Failure Reason':<20}")
        print("-" * 80)

        for result in hard_envs[:10]:
            # Find the main non-success status
            non_success_stats = {k: v for k, v in result['status_stats'].items()
                                 if k != 'success'}
            if non_success_stats:
                main_failure = max(non_success_stats.items(),
                                   key=lambda x: x[1]['count'])[0]
            else:
                main_failure = "unknown"

            print(
                f"{result['world_id']:<8} {result['success_rate']:<7.1f}% {result['avg_time']:<9.3f}s {main_failure:<20}")

    print("=" * 80)

if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    baseline_dir = "data/ddp_param-v0/test/"    # Directory containing baseline CSV files
    top_n_per_env = 20                  # Number of episodes per environment to analyze
    GOAL = 15                           # Goal position Y coordinate (10 or 15)
    
    # ============================================================================
    # MAIN ANALYSIS
    # ============================================================================
    print("🚀 Starting BARN Challenge Environment Analysis")
    print(f"📁 Baseline directory: {baseline_dir}")
    print(f"📊 Episodes per environment: {top_n_per_env}")
    print(f"🎯 Goal configuration: {GOAL}")
    print("=" * 80)
    
    # Analyze baseline results
    sorted_results = analyze_baseline_results(baseline_dir, top_n_per_env)
    
    # Generate detailed statistics
    analyze_status_distribution(sorted_results, top_n_per_env)
    
    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    suffix = f"_{GOAL}"
    
    # Save comprehensive results to CSV
    save_results(sorted_results, f"results/hard_environments{suffix}.csv")
    
    # Get and save hardest environments
    hardest_50 = get_hardest_environments(sorted_results, 50)
    
    with open(f"results/hardest_environments{suffix}.txt", "w") as f:
        for world_id in hardest_50:
            f.write(f"{world_id}, ")
    
    print(f"✅ Analysis complete! Results saved with suffix '{suffix}'")
    print(f"📄 Hardest 50 environments saved to: hardest_environments{suffix}.txt")