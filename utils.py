# -*- coding: utf-8 -*-

from pathlib import Path
import torch

CHECKPOINT_SAVE_PATH = Path('/tmp/checkpoint.pt')


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.min_loss = float("inf")
    
    def record_loss(self, loss, model):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
            torch.save(model.state_dict(), CHECKPOINT_SAVE_PATH)
        else:
            self.counter += 1
    
    def should_stop(self):
        return self.counter >= self.patience