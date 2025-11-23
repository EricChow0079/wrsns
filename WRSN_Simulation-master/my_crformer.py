# my_crformer.py   ←  新建这个文件，复制下面全部内容
import torch
import torch.nn as nn


class CRFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 16)  # 输入4维 → 16维
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
        # x: (B, 10, 4)
        x = self.proj(x)  # → (B, 10, 16)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.classifier(x).squeeze(-1)