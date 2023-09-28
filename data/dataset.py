import json
import time
import random
import hashlib

import torch
import numpy as np
import fortepyan as ff
from tqdm import tqdm
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset as TorchDataset

from data.tokenizer import MidiEncoder
from data.quantizer import MidiQuantizer
from data.augmentation import pitch_shift, change_speed


def apply_augmentation(record: dict, augmentation_probability: float):
    augmented = record.copy()
    notes = augmented["notes"]
    # check if augmentation happened
    done = False
    # shift pitch augmentation
    if random.random() < augmentation_probability:
        # max shift is octave down or up
        shift = random.randint(1, 12)
        notes["pitch"] = pitch_shift(notes["pitch"], shift)
        done = True

    # change tempo augmentation
    if random.random() < augmentation_probability:
        notes = change_speed(notes)
        done = True

    augmented["notes"] = notes

    # if no augmentation was done, return None
    return augmented if done else None


def build_translation_dataset(
    dataset: Dataset,
    dataset_cfg: DictConfig,
    augmentation_probability: float = 0.5,
    augmentation_rep: int = 1,
) -> Dataset:
    elapsed = 0
    # ~90s for giant midi
    quantizer = MidiQuantizer(
        n_dstart_bins=dataset_cfg.quantization.dstart,
        n_duration_bins=dataset_cfg.quantization.duration,
        n_velocity_bins=dataset_cfg.quantization.velocity,
    )

    quantized_pieces = []
    for it, record in tqdm(enumerate(dataset), total=len(dataset)):
        t = time.time()
        augmented = [
            apply_augmentation(
                record=record,
                augmentation_probability=augmentation_probability,
            )
            for _ in range(augmentation_rep)
        ]
        elapsed += time.time() - t
        # append augmented records if augmentation happened and original record
        records = [aug for aug in augmented if aug is not None] + [record]
        for new_record in records:
            piece = ff.MidiPiece.from_huggingface(new_record)
            qpiece = quantizer.inject_quantization_features(piece)

            # We want to get back to the original recording easily
            qpiece.source |= {"base_record_id": it, "dataset_name": dataset.info.dataset_name}
            quantized_pieces.append(qpiece)
    print(elapsed)

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
    df = piece.df
    for jt in range(n_samples):
        start = jt * sequence_step
        finish = start + sequence_len
        part = df.iloc[start:finish]

        sequence = {
            "pitch": part.pitch.astype("int16").values.T,
            # Quantized features
            "dstart_bin": part.dstart_bin.astype("int16").values.T,
            "duration_bin": part.duration_bin.astype("int16").values.T,
            "velocity_bin": part.velocity_bin.astype("int16").values.T,
            # Ground truth
            "start": part.start.values,
            "end": part.end.values,
            "duration": part.duration.values,
            "velocity": part.velocity.values,
            "source": json.dumps(piece.source),
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

        source_tokens_ids, target_tokens_ids = self.add_cls_token(source_tokens_ids, target_tokens_ids)

        out = {
            "source_token_ids": torch.tensor(source_tokens_ids, dtype=torch.int64),
            "target_token_ids": torch.tensor(target_tokens_ids, dtype=torch.int64),
        }
        return out

    def get_complete_record(self, idx: int) -> dict:
        # The usual token ids + everything we store
        out = self[idx] | self.dataset[idx]
        return out

    def add_cls_token(self, src_token_ids: list[int], tgt_token_ids: list[int]):
        cls_token_id = self.tgt_encoder.token_to_id["<CLS>"]
        src_token_ids.insert(0, cls_token_id)
        tgt_token_ids.insert(0, cls_token_id)

        return src_token_ids, tgt_token_ids


def load_cache_dataset(
    dataset_cfg: DictConfig,
    dataset_name: str,
    split: str,
    force_build: bool = False,
) -> Dataset:
    # Prepare caching hash
    config_hash = hashlib.sha256()
    config_string = json.dumps(OmegaConf.to_container(dataset_cfg)) + split + dataset_name
    config_hash.update(config_string.encode())
    config_hash = config_hash.hexdigest()

    dataset_cache_path = f"tmp/datasets/{config_hash}"

    if not force_build:
        try:
            translation_dataset = Dataset.load_from_disk(dataset_cache_path)
            return translation_dataset
        except Exception as e:
            print("Failed loading cached dataset:", e)

    print("Building translation dataset from", dataset_name, split)
    midi_dataset = load_dataset(dataset_name, split=split)

    # make sure dataset.info contains dataset_name - giant-midi-sustain appears to do not
    midi_dataset.info.dataset_name = dataset_name
    translation_dataset = build_translation_dataset(
        dataset=midi_dataset,
        dataset_cfg=dataset_cfg,
    )
    translation_dataset.save_to_disk(dataset_cache_path)

    return translation_dataset
