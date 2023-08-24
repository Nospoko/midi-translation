import os.path
import itertools

import torch
import fortepyan as ff
from tqdm import tqdm
from tokenizer import Tokenizer
from datasets import load_dataset
from prepare_dataset import process_record

from data.quantizer import MidiQuantizer


class TokenizedMidiTranslationDataset:
    def __init__(
        self,
        split: str = "train",
        n_dstart_bins: int = 3,
        n_duration_bins: int = 10,
        n_velocity_bins: int = 10,
        sequence_len: int = 128,
        device: str = "cpu",
    ):
        self.sequence_len = sequence_len
        self.device = device

        self.quantizer = MidiQuantizer(
            n_dstart_bins=n_dstart_bins,
            n_duration_bins=n_duration_bins,
            n_velocity_bins=n_velocity_bins,
        )

        self.tokenizer_src = Tokenizer(keys=["pitch", "dstart_bin"])
        self.tokenizer_tgt = Tokenizer(keys=["duration_bin", "velocity_bin"])

        self.dataset = load_dataset(path="roszcz/maestro-v1", split=split)

        self.src_vocab, self.tgt_vocab = self._build_vocab()

        self.processed_records = []
        self.samples = []

        self._build()

    def _build(self):
        self.processed_records = self.load_dataset()

        for record in tqdm(self.processed_records):
            src_tokens = self.tokenizer_src(record)
            tgt_tokens = self.tokenizer_tgt(record)

            src_processed = [self.src_vocab.index(token) for token in src_tokens]
            tgt_processed = [self.tgt_vocab.index(token) for token in tgt_tokens]

            bs_id = torch.tensor([0], device=self.device)  # <s> token id
            eos_id = torch.tensor([1], device=self.device)  # </s> token id
            src = torch.cat(
                [bs_id, torch.tensor(src_processed, dtype=torch.int64, device=self.device), eos_id],
                0,
            )
            tgt = torch.cat(
                [bs_id, torch.tensor(tgt_processed, dtype=torch.int64, device=self.device), eos_id],
                0,
            )

            self.samples.append((src, tgt))

    def load_dataset(self):
        path = (
            "dataset-"
            f"{self.quantizer.n_dstart_bins}-"
            f"{self.quantizer.n_duration_bins}-"
            f"{self.quantizer.n_velocity_bins}.pt"
        )
        if os.path.isfile(path):
            processed_records = torch.load(f=path)
        else:
            processed_records = self._build_dataset()
            torch.save(obj=self.processed_records, f=path)
        return processed_records

    def _build_dataset(self) -> list[dict]:
        processed_records = []
        for record in tqdm(self.dataset, total=self.dataset.num_rows):
            piece = ff.MidiPiece.from_huggingface(record)
            processed_record = process_record(piece, self.sequence_len, self.quantizer)

            processed_records += processed_record
        return processed_records

    def _build_vocab(self) -> tuple[list[str], list[str]]:
        vocab_src, vocab_tgt = ["<s>", "</s>"], ["<s>", "</s>"]

        # every combination of pitch + dstart
        product = itertools.product(range(21, 109), range(self.quantizer.n_dstart_bins))
        for pitch, dstart in product:
            key = f"{pitch}-{dstart}"
            vocab_src.append(key)

        # every combination of duration + velocity
        product = itertools.product(range(self.quantizer.n_duration_bins), range(self.quantizer.n_velocity_bins))
        for duration, velocity in product:
            key = f"{duration}-{velocity}"
            vocab_tgt.append(key)

        return vocab_src, vocab_tgt

    def __getitem__(self, idx: int):
        return self.samples[idx]


def main():
    dataset = TokenizedMidiTranslationDataset()

    record = dataset.processed_records[0]

    src_tokens = dataset.tokenizer_src(record)
    tgt_tokens = dataset.tokenizer_tgt(record)

    sample = dataset[0]

    print(f"Record: {record} \n" f"src_tokens: {src_tokens} \n" f"tgt_tokens: {tgt_tokens} \n" f"sample: {sample}")


if __name__ == "__main__":
    main()
