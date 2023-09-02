import os.path
import itertools

import torch
import pandas as pd
import fortepyan as ff
from tqdm import tqdm
from datasets import load_dataset

from data.quantizer import MidiQuantizer
from data.tokenizer import Tokenizer, VelocityTokenizer
from data.prepare_dataset import process_record, unprocessed_samples


class TokenizedMidiDataset:
    def __init__(
        self,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        split: str = "train",
        n_dstart_bins: int = 3,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
        sequence_len: int = 128,
        device: str = "cpu",
        samples_path: str = None,
    ):
        self.split = split
        self.sequence_len = sequence_len
        self.device = device

        self.quantizer = MidiQuantizer(
            n_dstart_bins=n_dstart_bins,
            n_duration_bins=n_duration_bins,
            n_velocity_bins=n_velocity_bins,
        )

        self.tokenizer_src = src_tokenizer
        self.tokenizer_tgt = tgt_tokenizer

        if samples_path is None:
            self.samples_path = (
                f"tmp/datasets/samples-"
                f"{self.quantizer.n_dstart_bins}-"
                f"{self.quantizer.n_duration_bins}-"
                f"{self.quantizer.n_velocity_bins}-"
                f"{self.split}-"
                f"{self.tokenizer_src.keys[0]}-{self.tokenizer_src.keys[1]}-vs-"
                f"{self.tokenizer_tgt.keys[0]}-{self.tokenizer_tgt.keys[1]}.pt"
            )
        else:
            self.samples_path = samples_path

        self.dataset = load_dataset(path="roszcz/maestro-v1", split=split)

        self.src_vocab, self.tgt_vocab = self.build_vocab()

        self.processed_records, self.unprocessed_records = self.load_dataset()
        self.samples = self.load_samples()

    def load_samples(self) -> list[tuple[list[int], list[int]]]:
        samples = []
        if os.path.isfile(self.samples_path):
            samples = torch.load(self.samples_path)
        else:
            pbar = tqdm(zip(self.processed_records, self.unprocessed_records), total=len(self.processed_records))

            print("Tokenizing ... ")
            for processed_record, unprocessed_record in pbar:
                src_tokens = self.tokenizer_src(processed_record)
                tgt_tokens = self.tokenizer_tgt(processed_record)

                src_processed = [self.src_vocab.index(token) for token in src_tokens]
                tgt_processed = [self.tgt_vocab.index(token) for token in tgt_tokens]

                src = torch.tensor(src_processed, dtype=torch.int64, device=self.device)
                tgt = torch.tensor(tgt_processed, dtype=torch.int64, device=self.device)

                samples.append((src, tgt))
            torch.save(samples, self.samples_path)
        return samples

    def load_dataset(self) -> tuple[list[dict], list[dict]]:
        path = (
            "tmp/datasets/dataset-"
            f"{self.quantizer.n_dstart_bins}-"
            f"{self.quantizer.n_duration_bins}-"
            f"{self.quantizer.n_velocity_bins}-"
            f"{self.sequence_len}-"
            f"{self.split}.pt"
        )
        if os.path.isfile(path):
            records = torch.load(path)
            processed_records = records["quantized"]
            unprocessed_records = records["not_quantized"]
        else:
            processed_records, unprocessed_records = self._build_dataset()
            torch.save(
                {
                    "quantized": processed_records,
                    "not_quantized": unprocessed_records,
                },
                f=path,
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

    def build_vocab(self) -> tuple[list[str], list[str]]:
        vocab_src, vocab_tgt = ["<s>", "<blank>", "</s>"], ["<s>", "<blank>", "</s>"]
        iterators = {
            "pitch": range(21, 109),
            "duration_bin": range(self.quantizer.n_duration_bins),
            "dstart_bin": range(self.quantizer.n_dstart_bins),
            "velocity_bin": range(self.quantizer.n_velocity_bins),
        }

        src_product = itertools.product(
            iterators[self.tokenizer_src.keys[0]],
            iterators[self.tokenizer_src.keys[1]],
        )

        tgt_product = itertools.product(
            iterators[self.tokenizer_tgt.keys[0]],
            iterators[self.tokenizer_tgt.keys[1]],
        )

        for val1, val2 in src_product:
            key = f"{val1}-{val2}"
            vocab_src.append(key)
        for val1, val2 in tgt_product:
            key = f"{val1}-{val2}"
            vocab_tgt.append(key)

        return vocab_src, vocab_tgt

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class BinsToVelocityDataset(TokenizedMidiDataset):
    def __init__(
        self,
        split: str = "train",
        n_dstart_bins: int = 3,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
        sequence_len: int = 128,
        device: str = "cpu",
    ):
        tokenizer_src = Tokenizer(keys=["pitch", "dstart_bin", "duration_bin", "velocity_bin"])
        tokenizer_tgt = VelocityTokenizer()

        samples_path = (
            f"tmp/datasets/samples-"
            f"{n_dstart_bins}-"
            f"{n_duration_bins}-"
            f"{n_velocity_bins}-"
            f"{split}-" 
            f"bins-to-vel.pt"
        )

        super().__init__(
            src_tokenizer=tokenizer_src,
            tgt_tokenizer=tokenizer_tgt,
            split=split,
            n_dstart_bins=n_dstart_bins,
            n_duration_bins=n_duration_bins,
            n_velocity_bins=n_velocity_bins,
            sequence_len=sequence_len,
            device=device,
            samples_path=samples_path,
        )

    def build_vocab(self) -> tuple[list[str], list[str]]:
        vocab_src, vocab_tgt = ["<s>", "<blank>", "</s>"], ["<s>", "<blank>", "</s>"]

        # every combination of pitch + dstart
        product = itertools.product(
            range(21, 109),
            range(self.quantizer.n_dstart_bins),
            range(self.quantizer.n_duration_bins),
            range(self.quantizer.n_velocity_bins),
        )
        for pitch, dstart, duration, velocity in product:
            key = f"{pitch}-{dstart}-{duration}-{velocity}"
            vocab_src.append(key)

        # every combination of duration + velocity
        for velocity in range(128):
            key = f"{velocity}"
            vocab_tgt.append(key)

        return vocab_src, vocab_tgt

    def load_samples(self) -> list[tuple[list[int], list[int]]]:
        samples = []
        if os.path.isfile(self.samples_path):
            samples = torch.load(self.samples_path)
        else:
            pbar = tqdm(zip(self.processed_records, self.unprocessed_records), total=len(self.processed_records))

            print("Tokenizing ... ")
            for processed_record, unprocessed_record in pbar:
                src_tokens = self.tokenizer_src(processed_record)
                tgt_tokens = self.tokenizer_tgt(unprocessed_record)

                src_processed = [self.src_vocab.index(token) for token in src_tokens]
                tgt_processed = [self.tgt_vocab.index(token) for token in tgt_tokens]

                src = torch.tensor(src_processed, dtype=torch.int64, device=self.device)
                tgt = torch.tensor(tgt_processed, dtype=torch.int64, device=self.device)

                samples.append((src, tgt))
            torch.save(samples, self.samples_path)
        return samples


def main():
    src_tokenizer = Tokenizer(keys=["pitch", "dstart_bin"])
    tgt_tokenizer = Tokenizer(keys=["duration_bin", "velocity_bin"])
    dataset = TokenizedMidiDataset(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        split="validation",
        n_dstart_bins=3,
        n_duration_bins=3,
        n_velocity_bins=3,
    )

    unprocessed_record = dataset.unprocessed_records[0]
    processed_record = dataset.processed_records[0]
    processed_df = pd.DataFrame(processed_record)

    quantized_record = dataset.quantizer.apply_quantization(processed_df)
    quantized_record.pop("midi_filename")

    src_tokens = dataset.tokenizer_src(processed_df)
    tgt_tokens = dataset.tokenizer_tgt(processed_df)

    src_untokenized = dataset.tokenizer_src.untokenize(src_tokens)

    sample = dataset[0]

    print(
        f"Unprocessed record: \n {unprocessed_record} \n "
        f"Quantized record: \n {quantized_record}"
        f"Processed record: {processed_record} \n"
        f"untokenized record: {src_untokenized} \n"
        f"src_tokens: {src_tokens} \n"
        f"tgt_tokens: {tgt_tokens} \n"
        f"sample: {sample}"
    )


if __name__ == "__main__":
    main()
