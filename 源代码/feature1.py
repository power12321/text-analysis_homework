import pandas as pd

# 读取 CSV 文件
file_path = 'news_data.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 筛选 Symbol 为 688981 的数据
df_filtered = df[df['Symbol'] == 688981]

# 显示筛选后的内容
print(df_filtered)

# 如果需要，可以将筛选后的数据保存为新的 CSV 文件
df_filtered.to_csv('filtered_news_data.csv', index=False)
# 读取 CSV 和 Excel 文件
filtered_file_path = 'filtered_news_data.csv'  # 替换为你的 CSV 文件路径


# 读取 CSV 文件
file_path = 'filtered_news_data.csv'  # 替换为实际路径
df_filtered = pd.read_csv(file_path)

# 转换 DeclareDate 列为日期格式（如果需要）
df_filtered['DeclareDate'] = pd.to_datetime(df_filtered['DeclareDate'])

# 按 DeclareDate 分组并计算 SentimentScore 的平均值
df_grouped = df_filtered.groupby('DeclareDate', as_index=False).agg({
    'Symbol': 'first',  # 保留第一个 Symbol
    'SentimentScore': 'mean'  # 计算 SentimentScore 的平均值
})

# 显示合并后的数据
print(df_grouped)

# 如果需要，可以将合并后的数据保存为新的 CSV 文件
df_grouped.to_csv('grouped_news_data.csv', index=False)


stock_file_path = 'stock.xlsx'  # 替换为你的 Excel 文件路径

# 读取 CSV 文件
df_filtered = pd.read_csv('grouped_news_data.csv')

# 读取 Excel 文件
df_stock = pd.read_excel(stock_file_path)

# 合并数据集，以 stock.xlsx 为基础，使用 filtered_news_data.csv 来合并
df_merged = pd.merge(df_stock, df_filtered, left_on='交易日期', right_on='DeclareDate', how='left')
df_merged = df_merged[['交易日期', '日收盘价', 'SentimentScore']]
# 显示合并后的数据
print(df_merged)

# 如果需要，可以将合并后的数据保存为新的 CSV 文件
df_merged.to_csv('merged_data.csv', index=False)


