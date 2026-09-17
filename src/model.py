import src
import torch
import torch.nn as nn
from .transformer import Transformer
from .embedding import Embedding
from .mask import generate_mask


class SRFTM(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)

        self.transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Только embedding + scale. Позиции добавятся внутри encoder/decoder.
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)

        out = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.output_projection(out)

    @torch.no_grad()
    def greedy_decode(self, src, sos_id, eos_id, max_len=128, repetition_penalty=1.2):
        self.eval()
        batch_size = src.size(0)
        device = src.device

        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        encoder_output = self.transformer.encoder(src_emb)

        tgt = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
            mask = generate_mask(tgt.size(1)).to(device)

            out = self.transformer.decoder(tgt_emb, mask, encoder_output)
            logits = self.output_projection(out[:, -1, :])

            # Repetition penalty
            for b in range(batch_size):
                for prev_token in set(tgt[b].tolist()):
                    if prev_token in (sos_id, eos_id):
                        continue
                    if logits[b, prev_token] > 0:
                        logits[b, prev_token] /= repetition_penalty
                    else:
                        logits[b, prev_token] *= repetition_penalty

            next_token = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return tgt