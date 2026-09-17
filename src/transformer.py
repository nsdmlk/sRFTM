import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_encoder_layers, n_decoder_layers, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_encoder_layers, dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, d_ff, n_decoder_layers, dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, tgt_mask, encoder_output)
        return decoder_output