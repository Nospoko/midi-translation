import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size: int, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return self.criterion(x, true_dist)
