import pickle
from pathlib import Path
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

with open('models/srftm.pkl', 'rb') as f:
    _data = pickle.load(f)
    _model = _data['model']
    _tok = _data['tokenizer']
    _build_template = _data['build_template']
_model.eval()


@app.route('/formula/convert', methods=['POST'])
def convert():
    text = request.json.get('text', '').strip()
    if not text:
        return jsonify({'error': 'empty'}), 400

    template = _build_template(text)
    if template:
        return jsonify({'latex': template, 'source': 'template'})

    with torch.no_grad():
        src = torch.tensor([_tok.encode(text)])
        out = _model.greedy_decode(src, _tok.sos_id, _tok.eos_id, max_len=100, repetition_penalty=1.3)
        latex = _tok.decode(out[0].tolist(), skip_special=True)

    return jsonify({'latex': latex, 'source': 'model'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5090)