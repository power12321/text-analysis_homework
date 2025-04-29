import pandas as pd

# 读取2020和2021年的数据，提取所需列'DeclareDate', 'Symbol', 'NewsContent'
files_2020_2021 = ['新闻证券2020.csv', '新闻证券2021.csv']  #
dfs = []
for file in files_2020_2021:
    try:
        df = pd.read_csv(file, usecols=['DeclareDate', 'Symbol', 'NewsContent'], encoding='gbk')
        dfs.append(df)
    except UnicodeDecodeError:
        df = pd.read_csv(file, usecols=['DeclareDate', 'Symbol', 'NewsContent'], encoding='utf-8')
        dfs.append(df)
combined_20_21 = pd.concat(dfs, ignore_index=True)

# 读取2022年的数据
try:
    df_2022 = pd.read_csv('新闻证券2022.csv', usecols=['DeclareDate', 'Symbol', 'NewsContent'], encoding='gbk')
except UnicodeDecodeError:
    df_2022 = pd.read_csv('新闻证券2022.csv', usecols=['DeclareDate', 'Symbol', 'NewsContent'], encoding='utf-8')

# 保存结果
combined_20_21.to_csv('2020_2021data.csv', index=False, encoding='utf-8-sig')
df_2022.to_csv('2022data.csv', index=False, encoding='utf-8-sig')

print("处理完成！数据已保存为2020_2021data.csv和2022data.csv")