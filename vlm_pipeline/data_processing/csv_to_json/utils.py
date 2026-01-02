import pandas as pd


def get_row_from_trajectory(
    data_trajectory: str,
    FILES,
    enable_guardrail_a: bool = True,
    max_steps_per_trajectory: int = None,  # Filter trajectories with > max_steps
):

    base = pd.read_csv(FILES)

    df = pd.read_csv(data_trajectory)

    # Extract world number from World column (e.g., "world_0.world" → 0)
    # Then map to actor key in base (e.g., 0 → "actor_0") to get metrics
    world_info = None
    env_type = None  # 'good', 'mid', 'bad'
    max_trajectories = None  # Guardrail A: 护栏 A 上限

    if 'World' in df.columns and not df.empty:
        # Get the first World value (assuming all rows in same file have same world)
        world_str = str(df['World'].iloc[0])  # e.g., "world_0.world"

        # Extract the number: "world_0.world" → "0"
        import re
        match = re.search(r'world_(\d+)', world_str)
        if match:
            world_num = match.group(1)  # "0"
            actor_key = f"actor_{world_num}"  # "actor_0"

            # Check if this actor exists in base (difficulty_map.csv)
            if actor_key in base['key'].values:
                row = base.loc[base['key'] == actor_key].iloc[0]
                score = row['score']

                world_info = {
                    'actor_key': actor_key,
                    'score': score,
                    'avg_time': row['avg_time'],
                    'count': row['count']
                }

                # Determine environment type based on score (4-tier)
                if score >= 0.45:  # ≥ 90% of 0.5
                    env_type = 'good'
                    max_trajectories = 90
                elif score >= 0.35:  # 70-90% of 0.5
                    env_type = 'mid-good'
                    max_trajectories = 80
                elif score >= 0.25:  # 50-70% of 0.5
                    env_type = 'mid-bad'
                    max_trajectories = 70
                else:  # < 50% of 0.5
                    env_type = 'bad'
                    max_trajectories = 50

                print(f"[INFO] World: {world_str} → {actor_key}, Type: {env_type.upper()}, Score: {score:.4f}, Max: {max_trajectories}")

            else:
                print(f"[WARN] {actor_key} not found in difficulty_map.csv")
        else:
            print(f"[WARN] Could not extract world number from: {world_str}")

    # 1) Deduplicate BEFORE any sorting to ensure "keep='last'" means latest in file
    keys = [k for k in ['Method', 'World', 'Start_frame_id'] if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep='last').reset_index(drop=True)
    elif 'Start_frame_id' in df.columns:
        df = df.drop_duplicates(subset='Start_frame_id', keep='last').reset_index(drop=True)

    # 2) Numeric formatting
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].round(4)

    # 3) Filter trajectories by step count EARLY (before quality filtering)
    if 'Start_frame_id' in df.columns and 'Done_frame_id' in df.columns:
        # Calculate steps for each trajectory
        df['num_steps'] = df['Done_frame_id'] - df['Start_frame_id']

        # First: filter by median
        median_steps = df['num_steps'].median()
        print(f"[STATS] Steps per trajectory - Median: {median_steps:.1f}, Mean: {df['num_steps'].mean():.1f}, Min: {df['num_steps'].min()}, Max: {df['num_steps'].max()}")

        before_median = len(df)
        df = df[df['num_steps'] <= median_steps].copy()
        after_median = len(df)
        if before_median > after_median:
            print(f"[FILTER] Removed {before_median - after_median} trajectories with steps > median ({median_steps:.1f})")

        # Second: filter by max_steps_per_trajectory (if set)
        if max_steps_per_trajectory is not None:
            before_max = len(df)
            df = df[df['num_steps'] <= max_steps_per_trajectory].copy()
            after_max = len(df)
            if before_max > after_max:
                print(f"[FILTER] Removed {before_max - after_max} trajectories with steps > {max_steps_per_trajectory}")

        # Drop the temporary column
        df = df.drop(columns=['num_steps'])

    # 4) Sort for ranking (only if columns exist)
    if 'nav_metric' in df.columns and 'Time' in df.columns:
        df = df.sort_values(by=['nav_metric', 'Time'], ascending=[False, True]).reset_index(drop=True)

    # 5) Quality filter for bad environments
    if env_type == 'bad' and 'nav_metric' in df.columns:
        # Step 1: 优先选满分轨迹
        perfect = df[df['nav_metric'] == 0.5].copy()

        # Step 2: 如果满分不足 K，用高分补齐（>= 0.4）
        if len(perfect) < max_trajectories:
            high_quality = df[df['nav_metric'] >= 0.4].copy()
            df_filtered = high_quality
        else:
            df_filtered = perfect

        # Step 3: 如果高分也不足 1 条，跳过这个环境
        if df_filtered.empty:
            print(f"[SKIP] Bad env: no high-quality trajectories (nav_metric >= 0.4)")
            return pd.DataFrame(), world_info

        df = df_filtered.reset_index(drop=True)

    # 6) Take top K trajectories (up to max_trajectories)
    total = len(df)

    if enable_guardrail_a and max_trajectories is not None:
        k = min(total, max_trajectories)
    else:
        k = total

    k = max(1, k)
    result = df.iloc[:k].copy()

    return result, world_info
