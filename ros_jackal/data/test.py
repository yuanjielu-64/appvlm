import pandas as pd

df = pd.read_csv('ddp_qwen_15/data_trajectory_2000.csv')

# 提取world数字并筛选6的倍数
df['World'] = df['World'].str.extract(r'(\d+)').astype(int)
df_filtered = df.iloc[:]

# 处理Time: Collision=1 或 Status!=success 时设为50
df_filtered['Time_adjusted'] = df_filtered.apply(
    lambda row: 50 if (row['Collision'] == 1 or row['Status'] != 'success') else row['Time'],
    axis=1
)

# nav_metric统计
print(f"Average nav_metric: {df_filtered['nav_metric'].mean():.4f}")
print(f"Min: {df_filtered['nav_metric'].min():.4f}")
print(f"Max: {df_filtered['nav_metric'].max():.4f}")
print(f"Count: {len(df_filtered)}")

# 平均时间
print(f"\nAverage Time: {df_filtered['Time_adjusted'].mean():.4f}")

# Status百分比
status_pct = (df_filtered['Status'].value_counts() / len(df_filtered) * 100).round(4)
print("\nStatus分布:")
for status, pct in status_pct.items():
    print(f"{status}: {pct:.3f}%")