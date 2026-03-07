import os
import certifi

# 强制使用 certifi 的证书文件
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from model import InvestModel
from train import train_model
from predict import predict_signal

# ======================
# 股票配置（你可以随便改）
# ======================
STOCK_CODE = "0020.HK"
START_DATE = "2022-01-01"
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 获取数据（你要的原版风格）
# ======================
def fetch_data(stock_code, start, end):
    try:
        print(f"正在获取 {stock_code} 的数据...")
        data = yf.download(stock_code, start=start, end=end)
        if data.empty:
            print("数据为空！")
            return None
        
        # 处理 MultiIndex 列：去掉股票代码层，保留价格类型
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)  # 去掉第二层（股票代码）
        
        print(f"获取成功，共 {len(data)} 条")
        return data
    except Exception as e:
        print(f"出错: {e}")
        return None

# ======================
# 特征工程
# ======================
def add_features(df):
    # 确保 'Close' 是一维 Series（如果是 DataFrame 则压缩）
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].squeeze()
    
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_60'] = ta.trend.sma_indicator(df['Close'], window=60)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df = df.dropna()
    return df

def build_dataset(df, seq_len=30):
    feat_cols = ['Close', 'SMA_20', 'SMA_60', 'RSI', 'MACD']
    data = df[feat_cols].values

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i, :])
        y.append(1 if data_scaled[i,0] > data_scaled[i-1,0] else 0)

    return np.array(X), np.array(y), scaler 

def build_dataset(df, seq_len=30):
    feat_cols = ['Close', 'SMA_20', 'SMA_60', 'RSI', 'MACD']
    data = df[feat_cols].values

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i, :])
        # 预测明天的收盘价（缩放后的值）
        y.append(data_scaled[i, 0])   # 第 i 天的 Close（缩放后）

    return np.array(X), np.array(y), scaler

# ======================
# 执行
# ======================
df = fetch_data(STOCK_CODE, START_DATE, END_DATE)
if df is None:
    exit()

df = add_features(df)
X, y, scaler = build_dataset(df, SEQ_LEN)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 训练
model = InvestModel(input_size=5).to(DEVICE)
train_model(model, train_loader, EPOCHS, DEVICE)

# 预测
print("\n===== 今日投资建议 =====")
last_sequence = X[-1:]   # 缩放后的序列

# 从原始 DataFrame 中获取最新价格和均线
current_price = df['Close'].iloc[-1]
sma_20 = df['SMA_20'].iloc[-1]
sma_60 = df['SMA_60'].iloc[-1]

suggest, prob = predict_signal(model, last_sequence, DEVICE, current_price, sma_20, sma_60, scaler)
print(f"上涨概率: {prob:.2f}")
print(suggest)