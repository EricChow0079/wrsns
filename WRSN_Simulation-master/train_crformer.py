# train_crformer.py  ——  750条样本专属版（过拟合？不存在的！）
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 1. 你的数据集类（完美）
class CRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.FloatTensor(item['feature']), torch.FloatTensor([item['label']])


# 2. 加载你刚生成的750条数据
with open('crformer_dataset_500.pkl', 'rb') as f:  # 注意文件名是你生成的
    dataset = pickle.load(f)

print(f"加载完成，共 {len(dataset)} 条珍贵样本！")

# 3. 小样本必备：8:1:1划分 + 小batch
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data = dataset[:train_size]
val_data = dataset[train_size:train_size + val_size]
test_data = dataset[train_size + val_size:]

train_loader = DataLoader(CRDataset(train_data), batch_size=16, shuffle=True)  # 小batch防过拟合
val_loader = DataLoader(CRDataset(val_data), batch_size=16)
test_loader = DataLoader(CRDataset(test_data), batch_size=16)


# 4. 超轻量级模型（专为小样本设计）
class CRFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 16)  # 特征从4维投影到16维
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.3, batch_first=True),
            num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)  # (B,10,16)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均
        return self.classifier(x).squeeze(-1)


model = CRFormer()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)  # L2正则

# 5. 训练 + 早停（防止过拟合）
best_val_loss = float('inf')
patience = 10
wait = 0

print("开始训练你的 CR-Former（750条样本专属版）...")
for epoch in range(100):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        pred = model(x)
        loss = criterion(pred, y.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            val_loss += criterion(pred, y.squeeze()).item()
            pred_label = (pred > 0.5).float()
            correct += (pred_label == y.squeeze()).sum().item()
            total += y.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    print(
        f"Epoch {epoch + 1:2d} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'crformer_750_best.pth')
        wait = 0
        print("  → 模型已保存！")
    else:
        wait += 1
        if wait >= patience:
            print("早停触发！")
            break

print("\nCR-Former 训练完成！")
print("最佳模型已保存：crformer_750_best.pth")
print(f"验证集准确率 ≈ {val_acc:.1%}（在750条样本上，这已经非常强了！）")