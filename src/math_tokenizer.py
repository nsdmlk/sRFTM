import json


class MathTokenizer:
    def __init__(self):
        # Вокабуляр: специальные + цифры + буквы + символы + LaTeX
        self.special = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.chars = list(
            '0123456789'
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '+-*/=^_()[]{}<>,.;:!?|&%$#@~`\'"\\ \n\t'
            'αβγδεζηθικλμνξπρστυφχψω'
            'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ'
            '∑∏∫√∞≤≥≠≈→←↔∈∉⊂⊃∪∩'
        )
        # LaTeX-команды добавляются через multi-char tokens
        self.latex_commands = [
            '\\frac', '\\sqrt', '\\sum', '\\int', '\\prod',
            '\\alpha', '\\beta', '\\gamma', '\\delta', '\\theta',
            '\\pi', '\\sigma', '\\phi', '\\omega',
            '\\infty', '\\le', '\\ge', '\\ne', '\\to',
            '\\log', '\\ln', '\\exp', '\\sin', '\\cos', '\\tan',
            '\\hat', '\\bar', '\\vec',
        ]
        self.vocab = self.special + self.chars + self.latex_commands
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        self.pad_id = self.token_to_id['<pad>']
        self.sos_id = self.token_to_id['<sos>']
        self.eos_id = self.token_to_id['<eos>']
        self.unk_id = self.token_to_id['<unk>']

    def __len__(self):
        return len(self.vocab)

    def encode(self, text: str, max_len: int = 256) -> list[int]:
        # Сначала жадный поиск LaTeX-команд
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for cmd in self.latex_commands:
                if text.startswith(cmd, i):
                    tokens.append(cmd)
                    i += len(cmd)
                    matched = True
                    break
            if not matched:
                tokens.append(text[i])
                i += 1

        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_token.get(i, '') for i in ids)