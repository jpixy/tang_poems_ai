# model.py
import torch.nn as nn
from config import CONFIG


class PoetryModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        embed_dim = CONFIG.getint("Model", "embed_dim")
        hidden_dim = CONFIG.getint("Model", "hidden_dim")
        num_layers = CONFIG.getint("Model", "num_layers")

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
