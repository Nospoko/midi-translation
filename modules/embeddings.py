import math

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        return self.lut(x) * math.sqrt(self.d_model)
