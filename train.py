import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, loader, epochs, device):
    criterion = nn.MSELoss()   # 回归任务
    opt = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for e in range(epochs):
        total = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device).float()
            opt.zero_grad()
            pred = model(X)           # 形状 (batch,)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {e+1:2d} | Loss: {total/len(loader):.4f}")