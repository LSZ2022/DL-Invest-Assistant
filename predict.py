import torch
import numpy as np

def predict_signal(model, X_last, device, current_price, sma_20, sma_60, scaler):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_last, dtype=torch.float32).to(device)
        pred_scaled = model(X).item()   # 模型输出缩放后的预测收盘价

    # 逆变换：需要构造一个形状 (1,5) 的数组，将预测的 Close 放进去，其余列用0或其他值填充？
    # 更简单的方法：只对 Close 列进行逆变换，但 scaler 是基于所有5列的。
    # 我们可以用 scaler 的 mean_ 和 scale_ 来手动还原 Close 列，或者构造一个虚拟行。
    # 这里采用构造虚拟行的方法：
    dummy = np.zeros((1, 5))
    dummy[0, 0] = pred_scaled   # 预测的 Close 缩放值
    # 将 dummy 逆变换
    pred_original = scaler.inverse_transform(dummy)[0, 0]   # 还原后的预测收盘价

    diff = (pred_original - current_price) / current_price

    if diff > 0.02 and current_price > sma_20:
        return f"📈 建议买入，目标价 {pred_original:.2f} (+{diff*100:.1f}%)", diff
    elif diff < -0.02 and current_price < sma_60:
        return f"📉 建议卖出，止损价 {pred_original:.2f} ({diff*100:.1f}%)", diff
    else:
        return "⚪ 建议观望", diff