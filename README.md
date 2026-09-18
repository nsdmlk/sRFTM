# sRFTM — Small Rubium Formula Transformer Model

> A small transformer that converts informal math expressions into LaTeX.


<b>Small data · Small model · Small GPU</b><br>


---

## What is sRFTM?

**sRFTM** is a compact encoder-decoder transformer that transforms **informal mathematical notation** into **valid LaTeX**.

Write formulas the way you think them — the model does the rest.

| Input | Output |
|---|---|
| `x^2 = 9` | `x^{2} = 9` |
| `1/n` | `\frac{1}{n}` |
| `deriv(x^3)` | `\frac{d}{dx}(x^{3})` |
| `sum n=1 inf 1/n^2` | `\sum_{n=1}^{\infty} \frac{1}{n^{2}}` |
| `sqrt(a^2 + b^2)` | `\sqrt{a^{2} + b^{2}}` |

**Goal:** users should never need to write LaTeX manually.

---

## Why

- **~90% of students don't know LaTeX.**
- Writing formulas in Notion / Google Docs is painful.
- LaTeX syntax is a barrier that stops people from taking good notes.

sRFTM removes this barrier.

---

## Architecture

**Encoder-decoder transformer** (as in "Attention is All You Need"):

- Multi-Head Attention.
- Sinusoidal positional encoding.
- 2 encoder layers + 2 decoder layers.
- `d_model = 128`, `n_heads = 4`, `d_ff = 256`.
- **~1M parameters.**

**Core structure** (attention, encoder, decoder, positional encoding, mask) is taken from my previous project:

👉 [**LLM-Core**](https://github.com/nsdmlk/LLM-Core) — a transformer built from scratch for learning purposes.

sRFTM adapts that codebase for the specific task of formula → LaTeX conversion.

---

## Data

Training data is a JSONL file of `(input, output)` pairs:

```jsonl
{"input": "x^2", "output": "x^{2}"}
{"input": "1/n", "output": "\\frac{1}{n}"}
{"input": "deriv(x^3)", "output": "\\frac{d}{dx}(x^{3})"}
```

**Sources:**

- Generated via ChatGPT (multiple styles per formula).
- Manual curation.
- Augmentation (variable/number substitution).

**Categories covered:**

- Powers, fractions, roots.
- Derivatives, integrals, sums, limits.
- Greek letters, logarithms, trigonometry.
- Equations, known formulas, ML/statistics expressions.

---

## Training

- **Loss:** CrossEntropy with `<eos>` weight, label smoothing.
- **Optimizer:** AdamW with weight decay.
- **Scheduler:** Cosine Annealing.
- **Early stopping** on validation loss.
- **Batch size:** 16.
- **Epochs:** up to 100.

Trains on **Google Colab T4** in under an hour.

---

## Inference

Two decoding strategies:

- **Greedy** — fast, deterministic.
- **Beam search** — slower, higher quality.

Both support:

- `repetition_penalty` — reduces loops.
- `length_penalty` — encourages shorter outputs.

Example:

```python
from src.model import SRFTM
from src.math_tokenizer import MathTokenizer
import torch

tokenizer = MathTokenizer()
model = SRFTM(
    src_vocab_size=len(tokenizer),
    tgt_vocab_size=len(tokenizer),
).cuda()
model.load_state_dict(torch.load('srftm_best.pt'))
model.eval()

src = torch.tensor([tokenizer.encode("deriv(x^3)")], device='cuda')
out = model.beam_search(
    src, tokenizer.sos_id, tokenizer.eos_id,
    max_len=50, beam_width=5, length_penalty=0.6,
)
print(tokenizer.decode(out.tolist(), skip_special=True))
# \frac{d}{dx}(x^{3})
```

---
## License

**Apache 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

If you use sRFTM in your work, please keep the attribution in `NOTICE`.

---

## Credits

**Core architecture:** [LLM-Core](https://github.com/nsdmlk/LLM-Core) — my previous project.

**Built for:** [Rubium](https://rubium.tech) — educational platform for students.

---

<p align="center">
  <sub>Small Models · Big Impact</sub>
</p>
