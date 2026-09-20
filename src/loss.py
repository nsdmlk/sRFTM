# src/loss.py

import torch
import torch.nn as nn


class LengthWeightedLoss(nn.Module):
    """
    CrossEntropyLoss с весами, обратно пропорциональными длине target.
    Короткие примеры получают больше веса.
    """
    def __init__(self, pad_id, eos_id, vocab_size, eos_weight=5.0, label_smoothing=0.05, device='cuda'):
        super().__init__()
        self.pad_id = pad_id
        
        class_weights = torch.ones(vocab_size).to(device)
        class_weights[eos_id] = eos_weight
        
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction='none',
        )
    
    def forward(self, logits, tgt_output):
        """
        logits: [batch, seq, vocab]
        tgt_output: [batch, seq]
        """
        losses = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )
        losses = losses.view(tgt_output.size(0), tgt_output.size(1))
        
        # Длины
        lengths = (tgt_output != self.pad_id).sum(dim=1).float()
        lengths = lengths.clamp(min=1)
        
        # Веса
        weights = 1.0 / lengths
        weights = weights / weights.mean()
        
        loss_per_example = losses.mean(dim=1)
        loss = (loss_per_example * weights).mean()
        
        return loss