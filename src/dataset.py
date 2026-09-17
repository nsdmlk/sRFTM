# srftm/src/dataset.py

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SRFTMDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._load(path)

    def _load(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                data.append(item)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        src_ids = self.tokenizer.encode(item['input'], max_len=self.max_len)
        tgt_ids = self.tokenizer.encode(item['output'], max_len=self.max_len)

        # Добавляем <sos> и <eos> в target
        tgt_ids = [self.tokenizer.sos_id] + tgt_ids + [self.tokenizer.eos_id]

        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
        }


def collate_fn(batch, pad_id=0):
    max_src = max(x['src'].size(0) for x in batch)
    max_tgt = max(x['tgt'].size(0) for x in batch)

    src_padded = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)

    # 1 = разрешено, 0 = запрещено (pad)
    src_mask = torch.zeros(len(batch), 1, 1, max_src, dtype=torch.bool)

    for i, x in enumerate(batch):
        src_len = x['src'].size(0)
        tgt_len = x['tgt'].size(0)
        src_padded[i, :src_len] = x['src']
        tgt_padded[i, :tgt_len] = x['tgt']
        src_mask[i, 0, 0, :src_len] = True

    return src_padded, tgt_padded, src_mask