import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SRFTM
from src.math_tokenizer import MathTokenizer

tokenizer = MathTokenizer()
model = SRFTM(
    src_vocab_size=len(tokenizer), tgt_vocab_size=len(tokenizer),
    d_model=128, n_heads=4, d_ff=512,
    n_encoder_layers=2, n_decoder_layers=2,
)
model.load_state_dict(torch.load('models/srftm_v4_best.pt', map_location='cpu'))
model.eval()

class EncoderWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, src):
        src_emb = self.m.src_embedding(src) * (self.m.d_model ** 0.5)
        return self.m.transformer.encoder(src_emb)

class DecoderStep(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, tgt, encoder_output):
        tgt_emb = self.m.tgt_embedding(tgt) * (self.m.d_model ** 0.5)
        seq = tgt.size(1)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        out = self.m.transformer.decoder(tgt_emb, mask, encoder_output)
        logits = self.m.output_projection(out[:, -1, :])
        return logits

enc = torch.jit.trace(EncoderWrapper(model), torch.tensor([[1, 2, 3]]))
dec = torch.jit.trace(DecoderStep(model), (torch.tensor([[1, 2]]), torch.randn(1, 3, 128)))

enc.save('models/encoder.pt')
dec.save('models/decoder.pt')
print("Saved encoder.pt + decoder.pt")