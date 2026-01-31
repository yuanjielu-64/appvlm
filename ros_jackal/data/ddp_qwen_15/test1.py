import pandas as pd

df = pd.read_csv('data_trajectory.csv')

# 定义test_assignments中的world列表
test_assignments = [
    [0, 48, 96, 144, 192, 240, 288],     # Group 0
    [6, 54, 102, 150, 198, 246, 294],    # Group 1
    [12, 60, 108, 156, 204, 252],        # Group 2
    [18, 66, 114, 162, 210, 258],        # Group 3
    [24, 72, 120, 168, 216, 264],        # Group 4
    [30, 78, 126, 174, 222, 270],        # Group 5
    [36, 84, 132, 180, 228, 276],        # Group 6
    [42, 90, 138, 186, 234, 282],        # Group 7
]

# test_assignments = [
#     [144, 192, 240, 288],     # Group 0
#     [150, 198, 246, 294],    # Group 1
#     [108, 156, 204, 252],        # Group 2
#     [114, 162, 210, 258],        # Group 3
#     [120, 168, 216, 264],        # Group 4
#     [126, 174, 222, 270],        # Group 5
#     [132, 180, 228, 276],        # Group 6
#     [138, 186, 234, 282],        # Group 7
# ]

# 合并所有test world
all_test_worlds = []
for group in test_assignments:
    all_test_worlds.extend(group)

# 提取world数字
df['World'] = df['World'].str.extract(r'(\d+)').astype(int)

# 筛选test_assignments中的world
df_filtered = df[df['World'].isin(all_test_worlds)].copy()

# 处理Time: Collision=1 或 Status!=success 时设为50
df_filtered['Time_adjusted'] = df_filtered.apply(
    lambda row: 50 if (row['Collision'] == 1 or row['Status'] != 'success') else row['Time'],
    axis=1
)

# 总体统计
print("="*60)
print("Test Results (All Test Worlds)")
print("="*60)
print(f"Nav Metric:   {df_filtered['nav_metric'].mean():.4f}")
print(f"Avg Time:     {df_filtered['Time_adjusted'].mean():.2f}s")
print(f"Episodes:     {len(df_filtered)}")

# Success rate
success_count = (df_filtered['Status'] == 'success').sum()
success_rate = success_count / len(df_filtered) * 100
print(f"Success Rate: {success_rate:.1f}%")
print("="*60)
