from dataclasses import dataclass

import torch

from modules.encoderdecoder import subsequent_mask


@dataclass
class Record:
    src: torch.Tensor
    tgt: torch.Tensor
    src_mask: torch.Tensor


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor):
        self.src = src
        self.src_mask = torch.ones_like(src, dtype=torch.bool).unsqueeze(-2)

        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = self.make_std_mask(self.tgt)
        # count tokens
        self.ntokens = self.tgt_y.numel()

    def __rich_repr__(self):
        yield "Batch"
        yield "batch_size", len(self)
        yield "src", self.src.shape
        yield "tgt", self.tgt.shape
        yield "tgt_y", self.tgt_y.shape
        yield "ntokens", self.ntokens.item()

    @staticmethod
    def make_std_mask(tgt):
        """Create a mask to hide padding and future words."""
        tgt_mask = torch.ones_like(tgt, dtype=torch.bool).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    def to(self, device: torch.device):
        self.src = self.src.to(device)
        self.tgt = self.tgt.to(device)
        self.src_mask = self.src_mask.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        self.tgt_y = self.tgt_y.to(device)
        self.ntokens = self.ntokens.to(device)

    def __len__(self) -> int:
        return self.src.shape[0]

    def __getitem__(self, idx: int) -> Record:
        out = Record(
            src=self.src[idx],
            tgt=self.tgt[idx],
            src_mask=self.src_mask[idx],
        )
        return out
