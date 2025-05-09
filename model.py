# model.py
import torch.nn as nn
from config import CONFIG
import torch.nn.init as init


class PoetryModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # 从配置加载参数
        embed_dim = CONFIG.getint("Model", "embed_dim")
        hidden_dim = CONFIG.getint("Model", "hidden_dim")
        num_layers = CONFIG.getint("Model", "num_layers")
        dropout_rate = CONFIG.getfloat("Model", "dropout", fallback=0.3)

        # 网络结构
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,  # 仅在多层时启用dropout
            bidirectional=False,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """专业权重初始化方法"""
        # Embedding层使用均匀分布初始化
        init.uniform_(self.embedding.weight, -0.1, 0.1)

        # LSTM层使用正交初始化和零偏置
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                init.orthogonal_(param)
            elif "bias" in name:
                init.zeros_(param)
                # 设置遗忘门偏置为1（改善梯度流动）
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        # 全连接层使用Xavier初始化
        init.xavier_normal_(self.fc.weight)
        init.zeros_(self.fc.bias)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

