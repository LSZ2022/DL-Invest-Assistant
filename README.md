# DL-Invest-Assistant 深度學習投資輔助系統

基於深度學習（LSTM + 技術指標）的自動化股票分析與投資建議工具。

## 功能
- 自動抓取股票歷史數據
- 計算 SMA、RSI、MACD 等技術指標
- 深度學習預測漲跌趨勢
- 輸出：買入 / 賣出 / 觀望 建議

## 環境
- Python 3.10 ~ 3.13.3
- PyTorch

```bash
pip install -r requirements.txt
python main.py