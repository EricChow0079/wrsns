# crformer_model.py
import torch
import torch.nn as nn


class CRFormer(nn.Module):
    def __init__(self, feature_dim=6, T=10, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, T, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, T, feature_dim)
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)  # 或取CLS，也可以用最后一个时间步
        return self.classifier(x).squeeze(-1)