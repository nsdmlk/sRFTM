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

# Обёртка для trace (greedy_decode с фиксированными параметрами)
class TracedSRFTM(torch.nn.Module):
    def __init__(self, model, sos_id, eos_id):
        super().__init__()
        self.model = model
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, src):
        return self.model.greedy_decode(
            src, self.sos_id, self.eos_id,
            max_len=100, repetition_penalty=1.3,
        )

traced = TracedSRFTM(model, tokenizer.sos_id, tokenizer.eos_id)
traced.eval()

dummy = torch.tensor([[tokenizer.encode("x^2")[0]]])
ts = torch.jit.trace(traced, dummy)
ts.save('models/srftm_traced.pt')

# Также сохраним tokenizer отдельно
import pickle
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Saved models/srftm_traced.pt + tokenizer.pkl")