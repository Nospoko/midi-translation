import itertools

import numpy as np
import pandas as pd
from uaclient import yaml
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


class MidiEncoder:
    def __init__(self):
        self.token_to_id = None

    def tokenize(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *tokenize* implementation")

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Your encoder needs *untokenize* implementation")

    def decode(self, token_ids: list[int]) -> pd.DataFrame:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        df = self.untokenize(tokens)

        return df

    def encode(self, record: dict) -> list[int]:
        tokens = self.tokenize(record)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids


class QuantizedMidiEncoder(MidiEncoder):
    def __init__(self, quantization_cfg: DictConfig):
        self.quantization_cfg = quantization_cfg
        self.keys = ["pitch", "dstart_bin", "duration_bin", "velocity_bin"]
        self.specials = ["<s>", "</s>", "<blank>"]
        # Make a copy of special tokens ...
        self.vocab = list(self.specials)

        # ... and add midi tokens
        self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "QuantizedMidiEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self):
        src_iterators_product = itertools.product(
            # Always include 88 pitches
            range(21, 109),
            range(self.quantization_cfg.dstart),
            range(self.quantization_cfg.duration),
            range(self.quantization_cfg.velocity),
        )

        for pitch, dstart, duration, velocity in src_iterators_product:
            key = f"{pitch}-{dstart}-{duration}-{velocity}"
            self.vocab.append(key)

    def tokenize(self, record: dict) -> list[str]:
        # TODO I don't love the idea of adding tokens durint *tokenize* call
        # If we want to pretend that our midi sequences have start and finish
        # we should take care of that before we get here :alarm:
        tokens = ["<s>"]
        n_samples = len(record[self.keys[0]])
        for idx in range(n_samples):
            token = "-".join([f"{record[key][idx]:0.0f}" for key in self.keys])
            tokens.append(token)

        tokens.append("</s>")
        return tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        samples = []
        for token in tokens:
            if token in self.specials:
                continue

            values_txt = token.split("-")
            values = [eval(txt) for txt in values_txt]
            samples.append(values)

        df = pd.DataFrame(samples, columns=self.keys)

        return df


class VelocityEncoder(MidiEncoder):
    def __init__(self):
        self.key = "velocity"
        self.specials = ["<s>", "</s>", "<blank>"]

        # Make a copy of special tokens ...
        self.vocab = list(self.specials)

        # ... and add velocity tokens
        self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "VelocityEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self):
        self.vocab += [str(possible_velocity) for possible_velocity in range(128)]

    def tokenize(self, record: dict) -> list[str]:
        # TODO I don't love the idea of adding tokens durint *tokenize* call
        # If we want to pretend that our midi sequences have start and finish
        # we should take care of that before we get here :alarm:
        velocity_tokens = [str(velocity) for velocity in record["velocity"]]
        tokens = ["<s>"] + velocity_tokens + ["</s>"]
        return tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        velocities = [int(token) for token in tokens if token not in self.specials]
        df = pd.DataFrame(velocities, columns=["velocity"])

        return df


class DstartEncoder(MidiEncoder):
    def __init__(self, bins: int = 200):
        self.specials = ["<s>", "</s>", "<blank>"]
        self.bins = bins
        # Make a copy of special tokens ...
        self.vocab = list(self.specials)
        self.bin_edges = self.load_bin_edges()
        # ... and add velocity tokens
        self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def load_bin_edges(self):
        artifacts_path = to_absolute_path("artifacts/bin_edges.yaml")
        with open(artifacts_path, "r") as f:
            bin_edges = yaml.safe_load(f)

        dstart_bin_edges = bin_edges["dstart"][self.bins]
        return dstart_bin_edges

    def _build_vocab(self):
        self.vocab += [str(possible_bin) for possible_bin in range(self.bins)]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, record: dict) -> list[str]:
        # TODO I don't love the idea of adding tokens durint *tokenize* call
        # If we want to pretend that our midi sequences have start and finish
        # we should take care of that before we get here :alarm:

        df = pd.DataFrame(record)

        # Quantize only dstart
        df["next_start"] = df.start.shift(-1)
        df["dstart"] = df.next_start - df.start
        df["dstart_bin"] = np.digitize(df.dstart.fillna(0), self.bin_edges) - 1

        # get tokens from quantized data
        tokens = [str(dstart_bin) for dstart_bin in df["dstart_bin"]]

        tokens = ["<s>"] + tokens + ["</s>"]
        return tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        dstarts = [int(token) for token in tokens if token not in self.specials]
        df = pd.DataFrame(dstarts, columns=["dstart_bin"])

        return df
