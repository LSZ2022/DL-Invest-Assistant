import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import ta

def load_stock_data(ticker, period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df = df.dropna()

    # 技術指標
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_60'] = ta.trend.sma_indicator(df['Close'], window=60)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df = df.dropna()

    return df

def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, :])
        y.append(data[i, 0])  # 預測漲跌
    return np.array(X), np.array(y)

def build_dataset(df, seq_len=30):
    features = df[['Close', 'SMA_20', 'SMA_60', 'RSI', 'MACD']].values
    scaler = MinMaxScaler((-1, 1))
    scaled = scaler.fit_transform(features)

    X, y = create_sequences(scaled, seq_len)

    # 標籤：明天漲 = 1，跌 = 0
    y = (y > np.shift(y, 1)[seq_len:]).astype(float)
    return X, y, scaler