"""
prepare_dataset.py by Jasiek Kaczmarczyk
from https://github.com/Nospoko/midi-clip/blob/MIDI-44/midi-clip/data/prepare_dataset.py
"""
import os
import json

import fortepyan as ff
from tqdm import tqdm
from datasets import Value, Dataset, Features, Sequence, DatasetDict, load_dataset

from data.quantizer import MidiQuantizer


def process_dataset(dataset_path: str, split: str, sequence_len: int, quantizer: MidiQuantizer) -> list[dict]:
    dataset = load_dataset(dataset_path, split=split)

    processed_records = []
    for record in tqdm(dataset, total=dataset.num_rows):
        # print(record)
        piece = ff.MidiPiece.from_huggingface(record)

        processed_record = process_record(piece, sequence_len, quantizer)

        processed_records += processed_record

    return processed_records


def unprocessed_samples(piece: ff.MidiPiece, sequence_len: int) -> list[dict]:
    records = []
    midi_filename = piece.source["midi_filename"]
    step = sequence_len // 2
    df = piece.df.copy()
    for subset in df.rolling(window=sequence_len, step=step):
        # rolling sometimes creates subsets with shorter sequence length, they are filtered here
        if len(subset) != sequence_len:
            continue
        sequence = {
            "midi_filename": midi_filename,
            "pitch": subset.pitch.values,
            "start": subset.start.values,
            "end": subset.end.values,
            "duration": subset.duration.values,
            "velocity": subset.velocity.values,
        }
        records.append(sequence)
    return records


def process_record(piece: ff.MidiPiece, sequence_len: int, quantizer: MidiQuantizer) -> list[dict]:
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
        if "tgt_dstart_bin" in piece_quantized.df.keys():
            sequence |= {"tgt_dstart_bin": quantized.tgt_dstart_bin.astype("int16").values.T}

        record.append(sequence)
    return record


if __name__ == "__main__":
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # hyperparameters
    sequence_len = 128

    # That's where I'm downloading the LTAFDB data
    hf_dataset_path = "roszcz/maestro-v1"

    quantizer = MidiQuantizer(
        n_dstart_bins=7,
        n_duration_bins=7,
        n_velocity_bins=7,
    )

    train_records = process_dataset(hf_dataset_path, split="train", sequence_len=sequence_len, quantizer=quantizer)
    val_records = process_dataset(hf_dataset_path, split="validation", sequence_len=sequence_len, quantizer=quantizer)
    test_records = process_dataset(hf_dataset_path, split="test", sequence_len=sequence_len, quantizer=quantizer)

    # building huggingface dataset
    features = Features(
        {
            "midi_filename": Value(dtype="string"),
            "pitch": Sequence(feature=Value(dtype="int16"), length=sequence_len),
            "dstart_bin": Sequence(feature=Value(dtype="int16"), length=sequence_len),
            "duration_bin": Sequence(feature=Value(dtype="int16"), length=sequence_len),
            "velocity_bin": Sequence(feature=Value(dtype="int16"), length=sequence_len),
        }
    )

    # dataset = Dataset.from_list(records, features=features)
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, features=features),
            "validation": Dataset.from_list(val_records, features=features),
            "test": Dataset.from_list(test_records, features=features),
        }
    )

    dataset_id = f"{quantizer.n_dstart_bins}-{quantizer.n_duration_bins}-{quantizer.n_velocity_bins}"
    dataset.save_to_disk(dataset_dict_path=f"quantized_dataset-{dataset_id}")
