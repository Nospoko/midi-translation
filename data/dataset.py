import itertools

import torch
import pandas as pd
import fortepyan as ff
from tqdm import tqdm
from datasets import Dataset

from data.quantizer import MidiQuantizer
from data.prepare_dataset import process_record
from data.tokenizer import Tokenizer, VelocityTokenizer


class TokenizedMidiDataset:
    def __init__(
        self,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        dataset: Dataset,
        n_dstart_bins: int = 3,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
        sequence_len: int = 128,
    ):
        self.sequence_len = sequence_len
        self.n_dstart_bins = n_dstart_bins
        self.n_duration_bins = n_duration_bins
        self.n_velocity_bins = n_velocity_bins
        self.tokenizer_src = src_tokenizer
        self.tokenizer_tgt = tgt_tokenizer

        self.dataset = dataset

        self.quantizer = MidiQuantizer(
            n_dstart_bins=n_dstart_bins,
            n_duration_bins=n_duration_bins,
            n_velocity_bins=n_velocity_bins,
        )

        self.src_vocab, self.tgt_vocab = self.build_vocab()

        # Chopp into sequences
        self.records = self._build_dataset()

        # Convert into src-tgt pairs of tokens
        self.samples = self.load_samples()

    def __rich_repr__(self):
        yield "TokenizedMidiDataset"
        yield "size", len(self)
        yield "sequence_len", self.sequence_len
        yield "n_dstart_bins", self.n_dstart_bins
        yield "n_duration_bins", self.n_duration_bins
        yield "n_velocity_bins", self.n_velocity_bins
        yield "src_vocab", len(self.src_vocab)
        yield "tgt_vocab", len(self.tgt_vocab)

    def load_samples(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        samples = []

        pbar = tqdm(self.records, total=len(self.records))

        print("Tokenizing ... ")
        for processed_record in pbar:
            src_tokens = self.tokenizer_src(processed_record)
            tgt_tokens = self.tokenizer_tgt(processed_record)

            src_processed = [self.src_vocab.index(token) for token in src_tokens]
            tgt_processed = [self.tgt_vocab.index(token) for token in tgt_tokens]

            src = torch.tensor(src_processed, dtype=torch.int64)
            tgt = torch.tensor(tgt_processed, dtype=torch.int64)

            samples.append((src, tgt))

        return samples

    def _build_dataset(self) -> list[dict]:
        processed_records = []

        print("Building a dataset ...")
        for record in tqdm(self.dataset, total=self.dataset.num_rows):
            # print(record)
            piece = ff.MidiPiece.from_huggingface(record)

            processed_record = process_record(piece, self.sequence_len, self.quantizer)

            processed_records += processed_record

        return processed_records

    def build_vocab(self) -> tuple[list[str], list[str]]:
        vocab_src, vocab_tgt = ["<s>", "<blank>", "</s>"], ["<s>", "<blank>", "</s>"]
        iterators = {
            "pitch": range(21, 109),
            "duration_bin": range(self.quantizer.n_duration_bins),
            "dstart_bin": range(self.quantizer.n_dstart_bins),
            "velocity_bin": range(self.quantizer.n_velocity_bins),
        }

        src_product = itertools.product((iterators[key] for key in self.tokenizer_src.keys))

        tgt_product = itertools.product((iterators[key] for key in self.tokenizer_tgt.keys))

        for values in src_product:
            key = "-".join(values)
            vocab_src.append(key)
        for values in tgt_product:
            key = "-".join(values)
            vocab_tgt.append(key)

        return vocab_src, vocab_tgt

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class BinsToVelocityDataset(TokenizedMidiDataset):
    def __init__(
        self,
        dataset: Dataset,
        n_dstart_bins: int = 3,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
        sequence_len: int = 128,
    ):
        tokenizer_src = Tokenizer(keys=["pitch", "dstart_bin", "duration_bin", "velocity_bin"])
        tokenizer_tgt = VelocityTokenizer()

        super().__init__(
            src_tokenizer=tokenizer_src,
            tgt_tokenizer=tokenizer_tgt,
            dataset=dataset,
            n_dstart_bins=n_dstart_bins,
            n_duration_bins=n_duration_bins,
            n_velocity_bins=n_velocity_bins,
            sequence_len=sequence_len,
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


def main():
    src_tokenizer = Tokenizer(keys=["pitch", "dstart_bin"])
    tgt_tokenizer = Tokenizer(keys=["duration_bin", "velocity_bin"])
    from datasets import load_dataset

    hf_dataset = load_dataset("roszcz/maestro-v1", split="validation")
    dataset = TokenizedMidiDataset(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        dataset=hf_dataset,
        n_dstart_bins=5,
        n_duration_bins=3,
        n_velocity_bins=3,
    )

    processed_record = dataset.records[0]
    processed_df = pd.DataFrame(processed_record)

    quantized_record = dataset.quantizer.apply_quantization(processed_df)
    quantized_record.pop("midi_filename")

    src_tokens = dataset.tokenizer_src(processed_df)
    tgt_tokens = dataset.tokenizer_tgt(processed_df)

    src_untokenized = dataset.tokenizer_src.untokenize(src_tokens)

    sample = dataset[0]

    print(
        f"Quantized record: \n {quantized_record}"
        f"Processed record: {processed_record} \n"
        f"untokenized record: {src_untokenized} \n"
        f"src_tokens: {src_tokens} \n"
        f"tgt_tokens: {tgt_tokens} \n"
        f"sample: {sample}"
    )


if __name__ == "__main__":
    main()
