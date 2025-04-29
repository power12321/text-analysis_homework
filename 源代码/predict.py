import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 检测GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据
data = pd.read_csv('merged_data.csv', parse_dates=['交易日期'])
data.set_index('交易日期', inplace=True)
data.sort_index(inplace=True)

# 划分数据集
train_data = data['2020-07-16':'2022-06-06']
test_data = data['2022-06-06':'2022-12-30']

# 处理SentimentScore的缺失值
def process_sentiment(df, train_ref=None):
    df = df.copy()
    sentiment_values = df['SentimentScore'].values

    # 如果 train_ref 被提供，使用其最后的 SentimentScore 填充缺失值
    if train_ref is not None:
        df['SentimentScore'] = df['SentimentScore'].fillna(train_ref['SentimentScore'].iloc[-1])

    # 逐行处理缺失值
    for i in range(1, len(sentiment_values)):
        if np.isnan(sentiment_values[i]):
            # 如果当前值缺失
            if sentiment_values[i - 1] > 0.5:
                # 如果前一个值大于0.5，逐渐衰减至0.5
                sentiment_values[i] = max(sentiment_values[i - 1] - 0.003, 0.5)
            else:
                # 如果前一个值小于0.5，逐渐递增至0.5
                sentiment_values[i] = min(sentiment_values[i - 1] + 0.003, 0.5)
        else:
            # 如果当前值不缺失，更新前一个值，确保下一些缺失值能以当前值进行填充
            last_valid_value = sentiment_values[i]

    df['SentimentScore'] = sentiment_values
    return df

# 使用新的 process_sentiment 函数处理训练集和测试集
train_data = process_sentiment(train_data)
test_data = process_sentiment(test_data)
train_data.to_csv('1.csv', index=False)
test_data.to_csv('2.csv', index=False)

# 合并特征并标准化
features = ['日收盘价', 'SentimentScore']
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[features])
test_scaled = scaler.transform(test_data[features])

# 创建PyTorch数据集
class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.seq_length = seq_length
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length, 0]
        return x.to(device), y.to(device)

SEQ_LENGTH = 30
train_dataset = StockDataset(train_scaled, SEQ_LENGTH)
test_dataset = StockDataset(test_scaled, SEQ_LENGTH)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步
        return self.linear(out)

# 初始化模型
model = LSTMModel(input_size=2, hidden_size=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 4000
sentiment_weight = 0.035

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 预测函数
def predict_steps(model, initial_sequence, steps, test_sentiments, sentiment_weight):
    model.eval()
    current_sequence = initial_sequence.clone().detach()
    predictions = []

    with torch.no_grad():
        for i in range(steps):
            # 使用LSTM模型预测下一个股价
            output = model(current_sequence.unsqueeze(0))  # 扩展维度以符合批处理
            pred = output.item()  # 转换为Python标量

            # 获取当前的SentimentScore并确保是标量
            sentiment = test_sentiments[i].item() if isinstance(test_sentiments, np.ndarray) else test_sentiments[i]

            # 根据情感得分对股价预测进行调整
            if sentiment > 0.5:
                # 情感得分大于0.5时，股价上涨，涨幅与情感得分的差值相关
                pred += sentiment_weight * (sentiment - 0.5)  # 线性加权调整
            else:
                # 情感得分小于0.5时，股价下跌，跌幅与情感得分的差值相关
                pred -= sentiment_weight * (0.5 - sentiment)  # 线性加权调整

            # 创建新特征张量（预测结果和情感得分作为输入特征）
            new_features = torch.tensor(
                [[pred, float(sentiment)]],  # 显式转换为float
                dtype=torch.float32
            ).to(device)

            # 更新序列，将预测结果加入序列中
            current_sequence = torch.cat(
                [current_sequence[1:], new_features]
            )
            predictions.append(pred)  # 保存预测结果

    return np.array(predictions)

# 修改测试数据准备
initial_sequence = test_dataset.data[:SEQ_LENGTH].to(device)

# 通过 `.numpy()` 将 pandas Series 转换为 numpy 数组
test_sentiments = test_dataset.data[SEQ_LENGTH:, 1].cpu().numpy().flatten()  # 确保一维

# 执行预测
predicted_scaled = predict_steps(model, initial_sequence,
                                 len(test_sentiments),
                                 test_sentiments, sentiment_weight)

# 反标准化预测结果
dummy_matrix = np.zeros((len(predicted_scaled), 2))
dummy_matrix[:, 0] = predicted_scaled
dummy_matrix[:, 1] = test_sentiments
predicted_prices = scaler.inverse_transform(dummy_matrix)[:, 0]

# 获取真实价格
true_prices = test_data['日收盘价'].iloc[SEQ_LENGTH:]

# 确保 true_prices.index 是一维的 NumPy 数组
true_prices_index = true_prices.index.to_numpy()  # 将索引转换为 NumPy 数组

# 确保 predicted_prices 也是一维数组
predicted_prices = predicted_prices.flatten()

# 绘制结果
plt.figure(figsize=(14, 6))
plt.plot(true_prices_index, true_prices.values, label='True Price')  # 使用 .values 获取实际价格值
plt.plot(true_prices_index, predicted_prices, label='Predicted Price')
plt.title('Stock Price Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
