import pandas as pd

# 读取 CSV
df = pd.read_csv('actor_0/data_trajectory.csv')

# 计算 nav_metric 的平均值
avg_nav_metric = df['nav_metric'].mean()
print(f"Average nav_metric: {avg_nav_metric:.4f}")

# 其他统计信息
print(f"Min: {df['nav_metric'].min():.4f}")
print(f"Max: {df['nav_metric'].max():.4f}")
print(f"Std: {df['nav_metric'].std():.4f}")
print(f"Median: {df['nav_metric'].median():.4f}")