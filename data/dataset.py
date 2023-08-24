import os.path
import itertools

import pandas as pd
import torch
import fortepyan as ff
from tqdm import tqdm
from datasets import load_dataset

from data.tokenizer import Tokenizer
from data.prepare_dataset import process_record, unprocessed_samples
from data.quantizer import MidiQuantizer


class TokenizedMidiDataset:
    def __init__(
        self,
        split: str = "train",
        n_dstart_bins: int = 3,
        n_duration_bins: int = 10,
        n_velocity_bins: int = 10,
        sequence_len: int = 128,
        device: str = "cpu",
    ):
        self.split = split
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
        self.processed_records, self.unprocessed_records = self.load_dataset()

        print("Tokenizing ... ")
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
            f"{self.quantizer.n_velocity_bins}-"
            f"{self.split}.pt"
        )
        if os.path.isfile(path):
            records = torch.load(path)
            processed_records = records['quantized']
            unprocessed_records = records['not_quantized']
        else:
            processed_records, unprocessed_records = self._build_dataset()
            torch.save(
                {
                    "quantized": processed_records,
                    "not_quantized": unprocessed_records,
                },
                f=path
            )
        return processed_records, unprocessed_records

    def _build_dataset(self) -> tuple[list[dict], list[dict]]:

        processed_records = []
        unprocessed_records = []
        print("Building a dataset ...")
        for record in tqdm(self.dataset, total=self.dataset.num_rows):
            # print(record)
            piece = ff.MidiPiece.from_huggingface(record)

            processed_record = process_record(piece, self.sequence_len, self.quantizer)
            unprocessed_record = unprocessed_samples(piece, self.sequence_len)

            unprocessed_records += unprocessed_record
            processed_records += processed_record

        return processed_records, unprocessed_records

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

    def __len__(self):
        return len(self.samples)


def main():
    dataset = TokenizedMidiDataset()
    unprocessed_record = dataset.unprocessed_records[0]
    processed_record = dataset.processed_records[0]
    processed_df = pd.DataFrame(processed_record)

    quantized_record = dataset.quantizer.apply_quantization(processed_df)
    quantized_record.pop('midi_filename')

    print(quantized_record)

    src_tokens = dataset.tokenizer_src(processed_record)
    tgt_tokens = dataset.tokenizer_tgt(processed_record)

    sample = dataset[0]

    print(
        f"Unprocessed record: \n {unprocessed_record} \n "
        f"Quantized record: \n {quantized_record}" 
        f"Processed record: {processed_record} \n" 
        f"src_tokens: {src_tokens} \n" 
        f"tgt_tokens: {tgt_tokens} \n" 
        f"sample: {sample}"
    )


if __name__ == "__main__":
    main()
