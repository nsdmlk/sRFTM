import pickle
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import srftm_models
from srftm_models import SRFTM, MathTokenizer, build_template

tokenizer = MathTokenizer()
model = SRFTM(
    src_vocab_size=len(tokenizer), tgt_vocab_size=len(tokenizer),
    d_model=128, n_heads=4, d_ff=512,
    n_encoder_layers=2, n_decoder_layers=2,
)
model.load_state_dict(torch.load('models/srftm_v4_best.pt', map_location='cpu'))
model.eval()

with open('models/srftm.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'tokenizer': tokenizer,
        'build_template': build_template,
    }, f, protocol=4)

print("Saved")