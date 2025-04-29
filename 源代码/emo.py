import pandas as pd
import torch
from tqdm import tqdm  # 导入 tqdm 以显示进度条
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载中文BERT预训练模型和tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 将BERT模型移到GPU（如果有的话）
model = model.to(device)

# 定义与股票相关的中文关键词（与股票涨跌相关的词汇）
up_keywords = [
    '增长', '盈利', '业绩', '扩张', '收购', '创新', '突破', '发展', '市场需求', '投资', '股东回报', '利润',
    '利好', '好消息', '强劲', '飙升', '红利', '增长率', '回报', '股市繁荣', '开盘上涨',
    '并购', '盈利能力', '股东奖励', '业绩增长', '产品创新', '市场领先', '财务改善', '营收增长', '战略成功',
    '行业繁荣', '市场份额提升', '行业扩张', '需求增加', '行业前景', '市场前景', '经济复苏', '外资进入',
    '政策支持', '利率降低', '政策利好', '增长动力', '市场信心'
]

down_keywords = [
    '亏损', '裁员', '下滑', '跌幅', '亏损', '危机', '负面', '下降', '停滞', '下降', '萎缩', '负债', '风险',
    '报告不佳', '低迷', '倒闭', '萎靡', '亏损预期', '盈利下滑', '业绩亏损', '负债累累', '财务危机', '销售下降',
    '裁员计划', '经营困难', '产品召回', '行业衰退', '市场份额丧失', '需求下降', '行业疲软', '市场不景气',
    '经济衰退', '通货紧缩', '失业率上升', '政策收紧', '货币紧缩', '金融危机', '外资撤出', '股市暴跌'
]


# 合并上涨和下跌的关键词并去重
all_keywords = list(set(up_keywords + down_keywords))  # 使用set去除重复项

# 使用CountVectorizer提取关键词频率
vectorizer = CountVectorizer(vocabulary=all_keywords)

# 数据集类
class NewsDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx] if self.labels is not None else None

# 读取新闻数据
def process_news_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(subset=['NewsContent'], inplace=True)  # 去除包含空内容的新闻

    # 使用CountVectorizer提取关键词频率
    keyword_matrix = vectorizer.transform(data['NewsContent'])

    # 将关键词频率添加到数据框
    keyword_features = pd.DataFrame(keyword_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # 情感分析
    def split_text_into_chunks(text, max_length=512):
        max_chunk_length = max_length - 2  # 减去2个Token的位置给[CLS]和[SEP]
        tokens = tokenizer.tokenize(text)
        chunks = [tokens[i:i + max_chunk_length] for i in range(0, len(tokens), max_chunk_length)]
        return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    def analyze_sentiment_batch(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            sentiment_scores = torch.softmax(logits, dim=-1).cpu().numpy()
        return sentiment_scores[:, 1]  # 返回正面情感的概率

    # 批量处理
    def process_batches(data, batch_size=16):
        sentiments = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data.iloc[i:i + batch_size]
            text_batch = batch['NewsContent'].tolist()
            sentiment_batch = analyze_sentiment_batch(text_batch)
            sentiments.extend(sentiment_batch)
        return sentiments

    # 批量分析情感得分
    sentiments = process_batches(data)
    data['SentimentScore'] = sentiments

    # 使用up_keywords和down_keywords的频率对情感得分进行调整
    def adjust_sentiment_with_keywords(row):
        up_count = sum([row[key] for key in up_keywords if key in row])
        down_count = sum([row[key] for key in down_keywords if key in row])
        sentiment_score = row['SentimentScore']
        sentiment_adjusted_score = sentiment_score + 0.1 * (up_count - down_count)  # 增加正面关键词的影响，减少负面关键词的影响
        return sentiment_adjusted_score

    # 调整情感得分
    data['AdjustedSentimentScore'] = data.apply(adjust_sentiment_with_keywords, axis=1)

    # 根据调整后的情感得分确定情感标签（正面/负面/中立）
    def sentiment_label(score):
        if score > 0.5:
            return 'POSITIVE'
        elif score < 0.5:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'

    data['SentimentLabel'] = data['AdjustedSentimentScore'].apply(sentiment_label)

    # 生成情感倾向（Sentiment Tendency）- 正面1，负面-1，中立0
    def sentiment_tendency(label):
        if label == 'POSITIVE':
            return 1
        elif label == 'NEGATIVE':
            return -1
        else:
            return 0  # 中立

    data['SentimentTendency'] = data['SentimentLabel'].apply(sentiment_tendency)
    data['SentimentStrength'] = data['AdjustedSentimentScore']

    # 将DeclareDate和Symbol列添加到结果中
    data = pd.concat([data[['DeclareDate', 'Symbol']], keyword_features, data[['SentimentLabel', 'SentimentScore', 'SentimentTendency', 'SentimentStrength']]], axis=1)

    return data

# 处理2020_2021data.csv和2022data.csv，批量处理
data_2020_2021 = process_news_data('2020_2021data.csv')
data_2022 = process_news_data('2022data.csv')

# 合并数据
final_data = pd.concat([data_2020_2021, data_2022], ignore_index=True)

# 保存结果为CSV文件
final_data.to_csv('news_text_quantification.csv', index=False)

print("文本量化指标已保存到 news_text_quantification.csv")
