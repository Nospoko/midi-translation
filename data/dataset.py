import hashlib
import itertools
import json
import os
import pickle

import torch
import pandas as pd
import fortepyan as ff
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from datasets import Dataset, load_dataset

from data.prepare_dataset import process_record
from data.tokenizer import Tokenizer, VelocityTokenizer
from data.quantizer import MidiQuantizer, QuantizerWithDstart


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
        self.tokenizer_src = src_tokenizer
        self.tokenizer_tgt = tgt_tokenizer

        self.dataset = dataset

        self.quantizer = MidiQuantizer(
            n_dstart_bins=n_dstart_bins,
            n_duration_bins=n_duration_bins,
            n_velocity_bins=n_velocity_bins,
        )

        self.records = self._build_dataset()
        self.src_vocab, self.tgt_vocab = self.build_vocab()

        self.samples = self.load_samples()

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
            "pitch": [str(pitch) for pitch in range(21, 109)],
            "duration_bin": [str(dur) for dur in range(self.quantizer.n_duration_bins)],
            "dstart_bin": [str(ds) for ds in range(self.quantizer.n_dstart_bins)],
            "velocity_bin": [str(v) for v in range(self.quantizer.n_velocity_bins)],
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


class BinsToDstartDataset(TokenizedMidiDataset):
    def __init__(
        self,
        dataset: Dataset,
        n_dstart_bins: int = 3,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
        sequence_len: int = 128,
        n_tgt_dstart_bins: int = 100,
    ):
        self.n_tgt_dstart_bins = n_tgt_dstart_bins
        tokenizer_src = Tokenizer(keys=["pitch", "dstart_bin", "duration_bin", "velocity_bin"])
        tokenizer_tgt = Tokenizer(keys=["tgt_dstart_bin"])
        super().__init__(
            src_tokenizer=tokenizer_src,
            tgt_tokenizer=tokenizer_tgt,
            dataset=dataset,
            n_dstart_bins=n_dstart_bins,
            n_duration_bins=n_duration_bins,
            n_velocity_bins=n_velocity_bins,
            sequence_len=sequence_len,
        )

    def _build_dataset(self) -> list[dict]:
        processed_records = []

        print("Building a dataset ...")
        for record in tqdm(self.dataset, total=self.dataset.num_rows):
            # print(record)
            piece = ff.MidiPiece.from_huggingface(record)

            # quantizer attribute override
            self.quantizer = QuantizerWithDstart(
                n_dstart_bins=self.quantizer.n_dstart_bins,
                n_duration_bins=self.quantizer.n_duration_bins,
                n_velocity_bins=self.quantizer.n_velocity_bins,
                n_tgt_dstart_bins=self.n_tgt_dstart_bins,
            )

            processed_record = process_record(piece, self.sequence_len, self.quantizer)

            processed_records += processed_record

        return processed_records

    def build_vocab(self) -> tuple[list[str], list[str]]:
        vocab_src, vocab_tgt = ["<s>", "<blank>", "</s>"], ["<s>", "<blank>", "</s>"]
        iterators = {
            "pitch": [str(pitch) for pitch in range(21, 109)],
            "duration_bin": [str(dur) for dur in range(self.quantizer.n_duration_bins)],
            "dstart_bin": [str(ds) for ds in range(self.quantizer.n_dstart_bins)],
            "velocity_bin": [str(v) for v in range(self.quantizer.n_velocity_bins)],
            "tgt_dstart_bin": [str(tgt) for tgt in range(self.quantizer.n_tgt_dstart_bins)],
        }

        src_product = itertools.product(*[iterators[key] for key in self.tokenizer_src.keys])
        tgt_product = itertools.product(*[iterators[key] for key in self.tokenizer_tgt.keys])

        for values in src_product:
            key = "-".join(values)
            vocab_src.append(key)
        for values in tgt_product:
            key = "-".join(values)
            vocab_tgt.append(key)

        return vocab_src, vocab_tgt


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


def load_cached_dataset(
    cfg: DictConfig,
    split: str = "test",
) -> TokenizedMidiDataset:
    if cfg.dataset_class == "BinsToVelocityDataset":
        return load_velocity_dataset(cfg, split)
    elif cfg.dataset_class == "BinsToDstartDataset":
        return load_dstart_dataset(cfg, split)


def load_dstart_dataset(cfg, split):
    n_dstart_bins, n_duration_bins, n_velocity_bins = cfg.bins.split(" ")
    n_dstart_bins, n_duration_bins, n_velocity_bins = int(n_dstart_bins), int(n_duration_bins), int(n_velocity_bins)

    config_hash = hashlib.sha256()
    config_string = json.dumps(OmegaConf.to_container(cfg)) + split
    config_hash.update(config_string.encode())
    config_hash = config_hash.hexdigest()
    cache_dir = "tmp/datasets"
    print(f"Preparing dataset: {config_hash}")
    try:
        dataset_cache_file = f"{config_hash}.pkl"
        dataset_cache_path = os.path.join(cache_dir, dataset_cache_file)

        if os.path.exists(dataset_cache_path):
            file = open(dataset_cache_path, "rb")
            dataset = pickle.load(file)

        else:
            file = open(dataset_cache_path, "wb")
            hf_dataset = load_dataset(cfg.dataset_name, split=split)

            args = [hf_dataset, n_dstart_bins, n_velocity_bins, n_duration_bins, cfg.sequence_size]
            dataset = BinsToDstartDataset(
                dataset=hf_dataset,
                n_dstart_bins=n_dstart_bins,
                n_velocity_bins=n_velocity_bins,
                n_duration_bins=n_duration_bins,
                sequence_len=cfg.sequence_size,
                n_tgt_dstart_bins=cfg.tgt_bins,
            )
            pickle.dump(dataset, file)

        file.close()

    except (EOFError, ConnectionError, UnboundLocalError):
        file.close()
        os.remove(path=dataset_cache_path)
        dataset = load_cached_dataset(cfg, split)

    return dataset


def load_velocity_dataset(cfg, split):
    n_dstart_bins, n_duration_bins, n_velocity_bins = cfg.bins.split(" ")
    n_dstart_bins, n_duration_bins, n_velocity_bins = int(n_dstart_bins), int(n_duration_bins), int(n_velocity_bins)

    config_hash = hashlib.sha256()
    config_string = json.dumps(OmegaConf.to_container(cfg)) + split
    config_hash.update(config_string.encode())
    config_hash = config_hash.hexdigest()
    cache_dir = "tmp/datasets"
    print(f"Preparing dataset: {config_hash}")
    try:
        dataset_cache_file = f"{config_hash}.pkl"
        dataset_cache_path = os.path.join(cache_dir, dataset_cache_file)

        if os.path.exists(dataset_cache_path):
            file = open(dataset_cache_path, "rb")
            dataset = pickle.load(file)

        else:
            file = open(dataset_cache_path, "wb")
            hf_dataset = load_dataset(cfg.dataset_name, split=split)

            dataset = BinsToVelocityDataset(
                dataset=hf_dataset,
                n_dstart_bins=n_dstart_bins,
                n_velocity_bins=n_velocity_bins,
                n_duration_bins=n_duration_bins,
                sequence_len=cfg.sequence_size,
            )
            pickle.dump(dataset, file)

        file.close()

    except (EOFError, ConnectionError, UnboundLocalError):
        file.close()
        os.remove(path=dataset_cache_path)
        dataset = load_cached_dataset(cfg, split)

    return dataset

def main():
    from datasets import load_dataset

    hf_dataset = load_dataset("roszcz/maestro-v1", split="validation")
    dataset = BinsToDstartDataset(
        dataset=hf_dataset,
        n_dstart_bins=5,
        n_duration_bins=3,
        n_velocity_bins=3,
        n_tgt_dstart_bins=10,
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