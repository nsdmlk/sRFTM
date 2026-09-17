import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 1. Masked Self-Attention — смотрит только на предыдущие слова
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        
        # 2. Cross-Attention — смотрит на выход энкодера (пока None)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        
        # 3. Feed-Forward — обработка
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # 4. Нормализация
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, encoder_output=None):
        # 1. Masked Self-Attention + residual + norm
        attn_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. Cross-Attention + residual + norm
        if encoder_output is not None:
            cross_out = self.cross_attention(x, encoder_output, encoder_output)
            x = self.norm2(x + self.dropout(cross_out))
        
        # 3. Feed-Forward + residual + norm
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x