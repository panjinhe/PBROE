import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("data/IDX_Idxtrdmth.csv")

# 要筛选的指数代码
target_indices = ["000852", "000300"]

# 保留这些指数的数据
filtered_df = df[df["Indexcd"].isin(target_indices)]

# 输出结果
print(filtered_df)

# 如果需要保存结果到新文件
filtered_df.to_csv("data/benchmark_indices.csv", index=False)
