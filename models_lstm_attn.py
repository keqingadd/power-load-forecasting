# -*- coding: utf-8 -*-
"""LSTM + 加性注意力（论文主模型，支持多通道外生特征）"""
import torch
import torch.nn as nn


class LSTMAttention(nn.Module):
    """LSTM 编码 + Bahdanau 加性注意力 + 全连接输出多步预测。

    n_features   : 输入通道数（1 = 纯负荷；9 = 负荷+时段/星期/周末/节假日/温度）
    n_future_cov : 目标时刻已知的协变量数（用于解码器，h 步 × n_future_cov）
    forward 返回 (pred, alpha)：
      pred  : (B, horizon)
      alpha : (B, lookback) 注意力权重，可画热力图做可解释性分析
    """

    def __init__(self, n_features: int = 1, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, horizon: int = 1, n_future_cov: int = 0,
                 n_quantiles: int = 1):
        super().__init__()
        self.horizon = horizon
        self.n_future_cov = n_future_cov
        self.n_quantiles = n_quantiles
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden, hidden)      # W·h_t
        self.v = nn.Linear(hidden, 1, bias=False)  # v^T·tanh(...)
        in_dim = hidden + horizon * n_future_cov
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, horizon * n_quantiles)
        )

    def forward(self, x, fx=None):
        if x.dim() == 2:                             # (B, L) -> (B, L, 1)
            x = x.unsqueeze(-1)
        h, _ = self.lstm(x)                          # (B, L, H)
        score = self.v(torch.tanh(self.proj(h)))     # (B, L, 1)
        alpha = torch.softmax(score.squeeze(-1), dim=1)   # (B, L)
        context = (alpha.unsqueeze(-1) * h).sum(dim=1)    # (B, H)
        if self.n_future_cov > 0 and fx is not None:
            context = torch.cat([context, fx.reshape(fx.shape[0], -1)], dim=1)
        return self.fc(context), alpha


def build_loss(name: str = "mae", quantiles=(0.1, 0.5, 0.9)):
    """损失函数工厂：mse / mae / huber / quantile"""
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    if name == "quantile":
        qs = torch.tensor(list(quantiles), dtype=torch.float32)

        def pinball(pred, target):
            # pred: (B, horizon*Q)  target: (B, horizon)
            b = pred.shape[0]
            h = target.shape[1]
            q = len(qs)
            p = pred.reshape(b, h, q)
            t = target.reshape(b, h, 1)
            err = t - p
            loss_q = torch.max(qs.reshape(1, 1, q) * err, (qs.reshape(1, 1, q) - 1) * err)
            return loss_q.mean()

        return pinball
    raise ValueError(f"未知损失函数: {name}")
