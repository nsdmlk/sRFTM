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
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)

        out = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.output_projection(out)

    # =========================================================
    # Greedy decode
    # =========================================================

    @torch.no_grad()
    def greedy_decode(
        self,
        src,
        sos_id,
        eos_id,
        max_len=128,
        repetition_penalty=1.2,
        length_penalty=0.0,
    ):
        self.eval()
        batch_size = src.size(0)
        device = src.device

        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        encoder_output = self.transformer.encoder(src_emb)

        tgt = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)

        for step in range(max_len):
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

            # Length penalty
            if length_penalty > 0:
                logits = logits / (1.0 + length_penalty * step)

            next_token = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return tgt

    # =========================================================
    # Beam search
    # =========================================================

    @torch.no_grad()
    def beam_search(
        self,
        src,
        sos_id,
        eos_id,
        max_len=128,
        beam_width=5,
        length_penalty=0.6,
        repetition_penalty=1.3,
    ):
        """
        Beam search с GNMT length penalty + repetition penalty.
        Для батча размера 1.
        """
        self.eval()
        device = src.device

        # Encoder
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        encoder_output = self.transformer.encoder(src_emb)

        # Beam: list of (tokens_tensor [1, L], log_prob)
        beams = [(torch.tensor([[sos_id]], dtype=torch.long, device=device), 0.0)]
        completed = []

        for step in range(max_len):
            candidates = []

            for tokens, log_prob in beams:
                if tokens[0, -1].item() == eos_id:
                    completed.append((tokens, log_prob))
                    continue

                # Forward
                tgt_emb = self.tgt_embedding(tokens) * (self.d_model ** 0.5)
                mask = generate_mask(tokens.size(1)).to(device)
                out = self.transformer.decoder(tgt_emb, mask, encoder_output)
                logits = self.output_projection(out[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1)

                # Repetition penalty
                for prev_token in set(tokens[0].tolist()):
                    if prev_token in (sos_id, eos_id):
                        continue
                    if log_probs[0, prev_token] < 0:
                        log_probs[0, prev_token] *= repetition_penalty
                    else:
                        log_probs[0, prev_token] /= repetition_penalty

                # Top-k
                top_k_log_probs, top_k_ids = log_probs.topk(beam_width, dim=-1)

                for k in range(beam_width):
                    new_tokens = torch.cat(
                        [tokens, top_k_ids[:, k:k + 1]], dim=1
                    )
                    new_log_prob = log_prob + top_k_log_probs[0, k].item()
                    candidates.append((new_tokens, new_log_prob))

            if not candidates:
                break

            # GNMT length penalty
            def score(item):
                tokens, log_prob = item
                length = tokens.size(1)
                lp = ((5 + length) / 6) ** length_penalty
                return log_prob / lp

            candidates.sort(key=score, reverse=True)
            beams = candidates[:beam_width]

            if all(t[0, -1].item() == eos_id for t, _ in beams):
                break

        # Best
        all_results = completed + beams
        all_results.sort(key=score, reverse=True)
        best_tokens = all_results[0][0]

        return best_tokens.squeeze(0)