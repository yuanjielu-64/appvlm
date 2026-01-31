import pandas as pd
import numpy as np


def _diversity_sampling(df: pd.DataFrame, max_trajectories: int, env_type: str) -> pd.DataFrame:
    """
    基于多样性的智能采样策略

    核心思想:
    1. 简单环境 (easy): 大部分都是0.5分，学习价值低 → 少采样
    2. 困难环境 (hard/very_hard): 有各种分数，学习价值高 → 多采样，保留多样性

    采样策略:
    - 分层采样: 按 nav_metric 分层 (0.5, 0.4-0.5, 0.3-0.4, ..., 0-0.1)
    - 每层按比例采样，确保覆盖不同质量的轨迹
    """

    if len(df) <= max_trajectories:
        print(f"[SAMPLING] Total trajectories ({len(df)}) ≤ quota ({max_trajectories}), keeping all")
        return df.copy()

    # 定义分层区间
    # 策略：手动分配 bin，0 和 0.5 独立
    def assign_quality_bin(score):
        if score == 0.5:
            return '0.5'
        elif 0.4 <= score < 0.5:
            return '0.4-0.5'
        elif 0.3 <= score < 0.4:
            return '0.3-0.4'
        elif 0.2 <= score < 0.3:
            return '0.2-0.3'
        elif 0.1 <= score < 0.2:
            return '0.1-0.2'
        elif 0 < score < 0.1:
            return '0.0-0.1'
        else:  # score == 0
            return '0'

    df['quality_bin'] = df['nav_metric'].apply(assign_quality_bin)
    labels = ['0', '0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5']

    # 统计每层数量
    bin_counts = df['quality_bin'].value_counts().sort_index()
    print(f"[DIVERSITY] Quality distribution:")
    for bin_name, count in bin_counts.items():
        print(f"  {bin_name}: {count} trajectories")

    # 采样策略根据环境类型调整
    if env_type == 'easy':
        # 简单环境: 只学成功案例，不学失败
        # 原因: 简单环境的失败是异常，不是常态，学了反而干扰模型
        sampling_weights = {
            '0.5': 0.8,        # 80% 配额给满分（学习标准答案）
            '0.4-0.5': 0.2,    # 20% 给高分（学习次优解）
            '0.3-0.4': 0.0,    # 不学中等分（无意义）
            '0.2-0.3': 0.0,    # 不学低分（负面样本）
            '0.1-0.2': 0.0,
            '0.0-0.1': 0.0,
            '0': 0.0,          # 不学完全失败
        }
        print(f"  [策略] 简单环境只学习成功案例 (≥0.4)")
    elif env_type == 'medium':
        # 中等环境: 重点学习成功和接近成功的案例
        # 原因: 有一定难度，但主要学习"如何成功"，少量学习"接近成功但失败"
        sampling_weights = {
            '0.5': 0.7,        # 60% 满分（标准答案）
            '0.4-0.5': 0.2,    # 20% 高分（次优解）
            '0.3-0.4': 0.1,   # 15% 中等分（边界案例）
            '0.2-0.3': 0.0,   # 5% 低分（少量失败案例供对比）
            '0.1-0.2': 0.0,    # 不学极低分（太差，无意义）
            '0.0-0.1': 0.0,
            '0': 0.0,          # 不学完全失败
        }
        print(f"  [策略] 中等环境主要学习成功案例，少量边界案例")
    elif env_type == 'hard':
        # 困难环境: 更均衡，重视中低分（学习价值高）
        sampling_weights = {
            '0.5': 0.6,
            '0.4-0.5': 0.25,
            '0.3-0.4': 0.15,
            '0.2-0.3': 0.0,
            '0.1-0.2': 0.0,
            '0.0-0.1': 0.0,
            '0': 0.0,
        }
    else:  # very_hard
        # 极难环境: 自适应策略
        # 如果成功案例太少，动态调整权重，偏向实际存在的数据
        total_high_quality = len(df[df['nav_metric'] >= 0.3])  # 高质量数据数量

        if total_high_quality < 20:  # 成功案例极少 (<20条)
            # 策略: 几乎放弃高分要求，重点学习失败案例
            sampling_weights = {
                '0.5': 0.05,      # 有几条要几条
                '0.4-0.5': 0.1,   # 少量高分
                '0.3-0.4': 0.3,   # 主要学习中等质量
                '0.2-0.3': 0.3,   # 主要学习中低质量
                '0.1-0.2': 0.2,   # 学习失败案例
                '0.0-0.1': 0.05,  # 少量极端失败
                '0': 0.0,         # 不学完全失败
            }
            print(f"  [策略] 成功案例极少 ({total_high_quality}条), 启用失败案例学习模式")
        elif total_high_quality < 50:  # 成功案例较少 (20-50条)
            # 策略: 降低高分权重，平衡学习
            sampling_weights = {
                '0.5': 0.1,
                '0.4-0.5': 0.15,
                '0.3-0.4': 0.3,
                '0.2-0.3': 0.25,
                '0.1-0.2': 0.15,
                '0.0-0.1': 0.05,
                '0': 0.0,
            }
            print(f"  [策略] 成功案例较少 ({total_high_quality}条), 启用平衡学习模式")
        else:  # 成功案例充足 (≥50条)
            # 策略: 标准均衡采样
            sampling_weights = {
                '0.5': 0.15,
                '0.4-0.5': 0.2,
                '0.3-0.4': 0.25,
                '0.2-0.3': 0.2,
                '0.1-0.2': 0.15,
                '0.0-0.1': 0.05,
                '0': 0.0,
            }
            print(f"  [策略] 成功案例充足 ({total_high_quality}条), 启用标准均衡模式")

    # 计算每层应采样的数量
    sampled_dfs = []
    total_allocated = 0

    for bin_name in labels:
        weight = sampling_weights.get(bin_name, 0)
        target_count = int(max_trajectories * weight)

        bin_df = df[df['quality_bin'] == bin_name]
        actual_count = min(len(bin_df), target_count)

        if actual_count > 0:
            # 每层内部按 Time 排序（选最快的）
            bin_df_sorted = bin_df.sort_values('Time', ascending=True) if 'Time' in bin_df.columns else bin_df
            sampled_dfs.append(bin_df_sorted.iloc[:actual_count])
            total_allocated += actual_count
            print(f"  Sampling {actual_count}/{len(bin_df)} from bin {bin_name}")

    # 合并所有采样结果（处理空列表情况）
    if len(sampled_dfs) == 0:
        print(f"  [警告] 没有采样到任何数据，返回空 DataFrame")
        return pd.DataFrame()

    result = pd.concat(sampled_dfs, ignore_index=True)

    # 如果因为某些层数据不足导致总数 < max_trajectories，从剩余数据补充
    if len(result) < max_trajectories:
        remaining_quota = max_trajectories - len(result)
        gap_ratio = remaining_quota / max_trajectories  # 缺口比例

        # 从未被采样的数据中补充（按 nav_metric 降序，但时间随机以增加多样性）
        sampled_indices = result.index
        remaining_df = df[~df.index.isin(sampled_indices)].sort_values('nav_metric', ascending=False)

        if len(remaining_df) > 0:
            # 如果缺口太大 (>30%)，说明数据质量分布不均，不强行凑数
            if gap_ratio > 0.3:
                # 只补充一部分，保证数据质量
                fill_ratio = 0.5  # 只补充缺口的一半
                actual_quota = int(remaining_quota * fill_ratio)
                print(f"  [补充] 缺口过大 ({gap_ratio:.1%}), 只补充 {actual_quota}/{remaining_quota} 条以保证质量")
            else:
                # 缺口小，可以全部补充
                actual_quota = remaining_quota

            # 从高分数据中随机采样（而不是按 Time 排序），增加多样性
            # 策略：取前 2*actual_quota 的高分数据，然后随机选 actual_quota 个
            candidate_size = min(len(remaining_df), actual_quota * 2)
            candidates = remaining_df.iloc[:candidate_size]

            if len(candidates) <= actual_quota:
                additional_samples = candidates
            else:
                additional_samples = candidates.sample(n=actual_quota, random_state=42)

            result = pd.concat([result, additional_samples], ignore_index=True)
            avg_metric = additional_samples['nav_metric'].mean()
            print(f"  [补充] Added {len(additional_samples)} trajectories (avg nav_metric: {avg_metric:.3f})")
        else:
            print(f"  [警告] 无剩余数据可补充，最终 {len(result)}/{max_trajectories} 条")

    # 删除临时列
    result = result.drop(columns=['quality_bin'])

    print(f"[SAMPLING] Final: {len(result)}/{max_trajectories} trajectories selected")
    return result


def get_row_from_trajectory(
    data_trajectory: str,
    FILES,
    enable_guardrail_a: bool = True,
    max_steps_per_trajectory: int = None,  # Filter trajectories with > max_steps
    use_diversity_sampling: bool = True,  # 新策略: 基于多样性采样
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

                # 新策略: 基于学习价值的采样配额
                # 核心思想: 简单环境少采样，复杂环境多采样
                if score >= 0.45:  # 简单环境 (大部分都成功)
                    env_type = 'easy'
                    max_trajectories = 30  # 降低配额 (学习价值低)
                elif score >= 0.30:  # 中等环境 (有成功有失败)
                    env_type = 'medium'
                    max_trajectories = 100  # 中等配额
                elif score >= 0.15:  # 困难环境 (很多失败)
                    env_type = 'hard'
                    max_trajectories = 150  # 高配额 (学习价值高)
                else:  # 极难环境 (几乎都失败)
                    env_type = 'very_hard'
                    max_trajectories = 150  # 最高配额 (最有学习价值)

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

    # 3) Filter trajectories by step count (保留合理的筛选)
    if 'Start_frame_id' in df.columns and 'Done_frame_id' in df.columns:
        df['num_steps'] = df['Done_frame_id'] - df['Start_frame_id']

        print(f"[STATS] Steps - Median: {df['num_steps'].median():.1f}, Mean: {df['num_steps'].mean():.1f}, Min: {df['num_steps'].min()}, Max: {df['num_steps'].max()}")

        # 只使用 max_steps_per_trajectory 硬上限 (如果设置了)
        # 移除 median 筛选，因为它太武断
        if max_steps_per_trajectory is not None:
            before = len(df)
            df = df[df['num_steps'] <= max_steps_per_trajectory].copy()
            after = len(df)
            if before > after:
                print(f"[FILTER] Removed {before - after} trajectories with steps > {max_steps_per_trajectory}")

        # 保留 num_steps 用于后续多样性采样
        # df = df.drop(columns=['num_steps'])  # 先不删除

    # 4) Sort for ranking (only if columns exist)
    if 'nav_metric' in df.columns and 'Time' in df.columns:
        df = df.sort_values(by=['nav_metric', 'Time'], ascending=[False, True]).reset_index(drop=True)

    # 5) 新策略: 基于多样性的智能采样
    if use_diversity_sampling and 'nav_metric' in df.columns and enable_guardrail_a and max_trajectories is not None:
        result = _diversity_sampling(df, max_trajectories, env_type)

        # 如果多样性采样返回空 DataFrame，回退到旧策略
        if len(result) == 0:
            print(f"[警告] 多样性采样失败，回退到简单采样策略")
            total = len(df)
            if enable_guardrail_a and max_trajectories is not None:
                k = min(total, max_trajectories)
            else:
                k = total
            k = max(1, k) if total > 0 else 0
            result = df.iloc[:k].copy() if k > 0 else pd.DataFrame()
    else:
        # 旧策略: 简单取前 K
        total = len(df)
        if enable_guardrail_a and max_trajectories is not None:
            k = min(total, max_trajectories)
        else:
            k = total
        k = max(1, k) if total > 0 else 0
        result = df.iloc[:k].copy() if k > 0 else pd.DataFrame()

    # 清理临时列
    if len(result) > 0 and 'num_steps' in result.columns:
        result = result.drop(columns=['num_steps'])

    return result, world_info, env_type
