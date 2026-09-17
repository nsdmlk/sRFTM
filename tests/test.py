from src.model import SRFTM
from src.math_tokenizer import MathTokenizer
import torch

tok = MathTokenizer()
model = SRFTM(
    src_vocab_size=len(tok),
    tgt_vocab_size=len(tok),
    d_model=128,
    n_heads=4,
    d_ff=512,
    n_encoder_layers=2,
    n_decoder_layers=2,
)

src = torch.randint(0, len(tok), (2, 10))
tgt = torch.randint(0, len(tok), (2, 8))

out = model(src, tgt)
print("Output shape:", out.shape)  # [2, 8, vocab_size]

out = model.greedy_decode(src, tok.sos_id, tok.eos_id, max_len=20)
print("Decoded shape:", out.shape)  # [2, N]