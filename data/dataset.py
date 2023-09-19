import json
import hashlib

import torch
import fortepyan as ff
from tqdm import tqdm
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset as TorchDataset

from data.quantizer import MidiQuantizer
from data.tokenizer import MidiEncoder, DstartEncoder, VelocityEncoder, QuantizedMidiEncoder


def build_translation_dataset(
    dataset: Dataset,
    dataset_cfg: DictConfig,
) -> Dataset:
    # ~90s for giant midi
    quantizer = MidiQuantizer(
        n_dstart_bins=dataset_cfg.quantization.dstart,
        n_duration_bins=dataset_cfg.quantization.duration,
        n_velocity_bins=dataset_cfg.quantization.velocity,
    )

    quantized_pieces = []
    for it, record in tqdm(enumerate(dataset), total=len(dataset)):
        piece = ff.MidiPiece.from_huggingface(record)
        qpiece = quantizer.inject_quantization_features(piece)

        # We want to get back to the original recording easily
        qpiece.source |= {"base_record_id": it, "dataset_name": dataset.info.dataset_name}
        quantized_pieces.append(qpiece)

    # ~20min for giant midi
    chopped_sequences = []
    for it, piece in tqdm(enumerate(quantized_pieces), total=len(quantized_pieces)):
        chopped_sequences += quantized_piece_to_records(
            piece=piece,
            sequence_len=dataset_cfg.sequence_len,
            sequence_step=dataset_cfg.sequence_step,
        )

    new_dataset = Dataset.from_list(chopped_sequences)
    return new_dataset


def quantized_piece_to_records(piece: ff.MidiPiece, sequence_len: int, sequence_step: int):
    chopped_sequences = []
    n_samples = 1 + (piece.size - sequence_len) // sequence_step
    for jt in range(n_samples):
        start = jt * sequence_step
        finish = start + sequence_len
        part = piece[start:finish]

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
        dataset_cfg: DictConfig,
        src_encoder: MidiEncoder,
        tgt_encoder: MidiEncoder,
    ):
        self.dataset = dataset
        self.dataset_cfg = dataset_cfg
        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder

    def __len__(self) -> int:
        return len(self.dataset)

    def __rich_repr__(self):
        yield "MyTokenizedMidiDataset"
        yield "size", len(self)
        # Nicer print
        yield "cfg", OmegaConf.to_container(self.dataset_cfg)
        yield "src_encoder", self.src_encoder
        yield "tgt_encoder", self.tgt_encoder

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        source_tokens_ids = self.src_encoder.encode(record)
        target_tokens_ids = self.tgt_encoder.encode(record)

        out = {
            "source_token_ids": torch.tensor(source_tokens_ids, dtype=torch.int64),
            "target_token_ids": torch.tensor(target_tokens_ids, dtype=torch.int64),
        }
        return out

    def get_complete_record(self, idx: int) -> dict:
        # The usual token ids + everything we store
        out = self[idx] | self.dataset[idx]
        return out


def load_cache_dataset(
    dataset_cfg: DictConfig,
    dataset_name: str,
    split: str,
    force_build: bool = False,
    predict_column: str = "velocity",
    tgt_bins: int = 200,
) -> MyTokenizedMidiDataset:
    # Prepare caching hash
    config_hash = hashlib.sha256()
    config_string = json.dumps(OmegaConf.to_container(dataset_cfg)) + split + dataset_name
    config_hash.update(config_string.encode())
    config_hash = config_hash.hexdigest()

    # Prepare midi encoders
    src_encoder = QuantizedMidiEncoder(dataset_cfg.quantization)
    if predict_column == "velocity":
        tgt_encoder = VelocityEncoder()
    elif predict_column == "dstart":
        tgt_encoder = DstartEncoder(bins=tgt_bins)
    dataset_cache_path = f"tmp/datasets/{config_hash}"
    if not force_build:
        try:
            translation_dataset = Dataset.load_from_disk(dataset_cache_path)

            tokenized_dataset = MyTokenizedMidiDataset(
                dataset=translation_dataset,
                dataset_cfg=dataset_cfg,
                src_encoder=src_encoder,
                tgt_encoder=tgt_encoder,
            )
            return tokenized_dataset
        except Exception as e:
            print("Failed loading cached dataset:", e)

    print("Building translation dataset from", dataset_name, split)
    midi_dataset = load_dataset(dataset_name, split=split)
    translation_dataset = build_translation_dataset(
        dataset=midi_dataset,
        dataset_cfg=dataset_cfg,
    )
    translation_dataset.save_to_disk(dataset_cache_path)

    tokenized_dataset = MyTokenizedMidiDataset(
        dataset=translation_dataset,
        dataset_cfg=dataset_cfg,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )
    return tokenized_dataset


if __name__ == "__main__":
    cfg = {
        "sequence_len": 128,
        "sequence_step": 42,
        "quantization": {
            "duration": 3,
            "dstart": 3,
            "velocity": 3,
        },
    }

    dataset = load_cache_dataset(
        OmegaConf.create(cfg), dataset_name="roszcz/maestro-v1-sustain", split="validation", predict_column="dstart"
    )

    print(len(dataset[0]["source_token_ids"]))
    print(len(dataset[0]["source_token_ids"]))
