import itertools

import json
import torch
import pandas as pd
import fortepyan as ff
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from data.quantizer import MidiQuantizer
from data.tokenizer import Tokenizer, VelocityTokenizer


def process_record(
    piece: ff.MidiPiece,
    sequence_len: int,
    quantizer: MidiQuantizer,
) -> list[dict]:
    piece_quantized = quantizer.quantize_piece(piece)

    midi_filename = piece_quantized.source["midi_filename"]

    record = []
    step = sequence_len // 2
    iterator = zip(
        piece.df.rolling(window=sequence_len, step=step),
        piece_quantized.df.rolling(window=sequence_len, step=step),
    )

    for subset, quantized in iterator:
        # rolling sometimes creates subsets with shorter sequence length, they are filtered here
        if len(quantized) != sequence_len:
            continue

        sequence = {
            "midi_filename": midi_filename,
            "pitch": quantized.pitch.astype("int16").values.T,
            "dstart_bin": quantized.dstart_bin.astype("int16").values.T,
            "duration_bin": quantized.duration_bin.astype("int16").values.T,
            "velocity_bin": quantized.velocity_bin.astype("int16").values.T,
            "start": subset.start.values,
            "end": subset.end.values,
            "duration": subset.duration.values,
            "velocity": subset.velocity.values,
            "source": json.dumps(piece.source),
        }

        record.append(sequence)

    return record


def process_dataset(
    dataset: Dataset,
    quantizer: MidiQuantizer,
    sequence_len: int,
    sequence_step: int,
):
    # ~90s
    quantized_pieces = []
    for it, record in tqdm(enumerate(dataset), total=len(dataset)):
        piece = ff.MidiPiece.from_huggingface(record)
        qpiece = quantizer.inject_quantization_features(piece)
        # We want to get back to the original recording easily
        qpiece.source |= {"base_record_id": it, "dataset_name": dataset.info.dataset_name}
        quantized_pieces.append(qpiece)

    # ~20min
    chopped_sequences = []
    for it, piece in tqdm(enumerate(quantized_pieces), total=len(quantized_pieces)):
        n_samples = (piece.size - sequence_len) // sequence_step
        for jt in range(n_samples):
            sta = jt * sequence_step
            finish = sta + sequence_len
            part = piece[sta: finish]

            sequence = {
                "pitch": part.df.pitch.astype("int16").values.T,
                # Quantized features
                "dstart_bin": part.df.dstart_bin.astype("int16").values.T,
                "duration_bin": part.df.duration_bin.astype("int16").values.T,
                "velocity_bin": part.df.velocity_bin.astype("int16").values.T,
                # Ground truth
                "start": part.df.start.values,
                "end": part.df.end.values,
                "duration": part.df.duration.values,
                "velocity": part.df.velocity.values,
                "source": json.dumps(part.source),
            }
            chopped_sequences.append(sequence)

    return chopped_sequences


class MyTokenizedMidiDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
    ):
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        source_tokens = self.src_tokenizer(record)
        target_tokens = self.tgt_tokenizer(record)

        out = {
            "src": source_tokens,
            "tgt": target_tokens,
        }
        return out


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
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

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
            src_tokens = self.src_tokenizer(processed_record)
            tgt_tokens = self.tgt_tokenizer(processed_record)

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

        src_product = itertools.product((iterators[key] for key in self.src_tokenizer.keys))

        tgt_product = itertools.product((iterators[key] for key in self.tgt_tokenizer.keys))

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
        # This dataset class only creates a parent class we a pre-defined set of aguments
        # It may be better to separate dataset, tokenizer, and vocab :thinking:
        src_tokenizer = Tokenizer(keys=["pitch", "dstart_bin", "duration_bin", "velocity_bin"])
        tgt_tokenizer = VelocityTokenizer()

        super().__init__(
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            dataset=dataset,
            n_dstart_bins=n_dstart_bins,
            n_duration_bins=n_duration_bins,
            n_velocity_bins=n_velocity_bins,
            sequence_len=sequence_len,
        )

    def build_vocab(self) -> tuple[list[str], list[str]]:
        vocab_src, vocab_tgt = ["<s>", "<blank>", "</s>"], ["<s>", "<blank>", "</s>"]

        # every combination of pitch + dstart
        src_iterators_product = itertools.product(
            range(21, 109),
            range(self.quantizer.n_dstart_bins),
            range(self.quantizer.n_duration_bins),
            range(self.quantizer.n_velocity_bins),
        )
        for pitch, dstart, duration, velocity in src_iterators_product:
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

    src_tokens = dataset.src_tokenizer(processed_df)
    tgt_tokens = dataset.tgt_tokenizer(processed_df)

    src_untokenized = dataset.src_tokenizer.untokenize(src_tokens)

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
