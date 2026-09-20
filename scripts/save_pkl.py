import pickle
import torch
from pathlib import Path
import sys
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

with open('models/srftm.pkl', 'wb') as f:
    pickle.dump({'model': model, 'tokenizer': tokenizer}, f)

print("Saved models/srftm.pkl")