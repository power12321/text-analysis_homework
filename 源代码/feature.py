import pandas as pd

# 读取 CSV 文件
file_path = 'news_text_quantification.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 提取指定的列
columns_to_extract = ['DeclareDate', 'Symbol', 'SentimentScore']
df_extracted = df[columns_to_extract]

# 显示提取的内容
print(df_extracted)

# 如果需要，可以保存提取的数据为新的 CSV 文件
df_extracted.to_csv('news_data.csv', index=False)
