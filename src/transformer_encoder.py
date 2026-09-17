import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .encoder_layer import EncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, mask=None):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x