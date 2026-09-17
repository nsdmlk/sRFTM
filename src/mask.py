import torch


def generate_mask(seq_len):
    """Создаёт triangular mask: слово видит только прошлое."""
    return torch.tril(torch.ones(seq_len, seq_len))