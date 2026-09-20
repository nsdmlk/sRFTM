import pickle
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.math_tokenizer import MathTokenizer

tok = MathTokenizer()

# Сохраняем как JSON (без pickle)
data = {
    'vocab': tok.vocab,
    'special': tok.special,
    'chars': tok.chars,
    'latex_commands': tok.latex_commands,
    'pad_id': tok.pad_id,
    'sos_id': tok.sos_id,
    'eos_id': tok.eos_id,
    'unk_id': tok.unk_id,
}

with open('models/tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

print("Saved models/tokenizer.json")