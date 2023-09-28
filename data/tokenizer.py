import itertools

import yaml
import numpy as np
import pandas as pd
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
        super().__init__()
        self.quantization_cfg = quantization_cfg
        self.keys = ["pitch", "dstart_bin", "duration_bin", "velocity_bin"]
        self.specials = ["<CLS>"]

        self.vocab = list(self.specials)

        # add midi tokens to vocab
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
        tokens = []
        n_samples = len(record[self.keys[0]])
        for idx in range(n_samples):
            token = "-".join([f"{record[key][idx]:0.0f}" for key in self.keys])
            tokens.append(token)

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
        super().__init__()
        self.key = "velocity"
        self.specials = ["<CLS>"]
        self.vocab = list(self.specials)

        # add velocity tokens
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
        tokens = [str(velocity) for velocity in record["velocity"]]
        return tokens

    def untokenize(self, tokens: list[str]) -> list[int]:
        velocities = [int(token) for token in tokens if token not in self.specials]

        return velocities


class DstartEncoder(MidiEncoder):
    def __init__(self, n_bins: int = 200):
        super().__init__()
        self.specials = ["<CLS>"]
        self.bins = n_bins

        self.vocab = list(self.specials)
        # add dstart tokens
        self._build_vocab()
        self._bin_edges = self._load_bin_edges()
        self.bin_to_dstart = []
        self._build_dstart_decoder()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def _load_bin_edges(self):
        artifacts_path = to_absolute_path("artifacts/bin_edges.yaml")
        with open(artifacts_path, "r") as f:
            bin_edges = yaml.safe_load(f)

        dstart_bin_edges = bin_edges["dstart"][self.bins]
        return dstart_bin_edges

    def _build_vocab(self):
        self.vocab += [str(possible_bin) for possible_bin in range(self.bins)]

    def _build_dstart_decoder(self):
        self.bin_to_dstart = []
        for it in range(1, len(self._bin_edges)):
            dstart = (self._bin_edges[it - 1] + self._bin_edges[it]) / 2
            self.bin_to_dstart.append(dstart)

        last_dstart = 2 * self._bin_edges[-1]
        self.bin_to_dstart.append(last_dstart)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def unquantized_start(self, dstart_bins: np.array) -> np.array:
        quant_dstart = [self.bin_to_dstart[it] for it in dstart_bins]
        start = pd.Series(quant_dstart).cumsum().shift(1).fillna(0)

        return start

    def quantize(self, start: list[float]) -> list[int]:
        dstart = []
        for it in range(len(start) - 1):
            dstart.append(start[it + 1] - start[it])
        dstart.append(0)

        dstart_bins = np.digitize(dstart, self._bin_edges) - 1

        return dstart_bins

    def tokenize(self, record: dict) -> list[str]:
        dstart_bins = self.quantize(record["start"])

        # get tokens from quantized data
        tokens = [str(dstart_bin) for dstart_bin in dstart_bins]
        return tokens

    def untokenize(self, tokens: list[str]) -> list[int]:
        dstarts = [int(token) for token in tokens if token not in self.specials]

        return dstarts
