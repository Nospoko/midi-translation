import random

import numpy as np
from tqdm import tqdm
from datasets import Dataset
from omegaconf import DictConfig

from data.quantizer import MidiQuantizer


def change_speed(record: dict, factor: float = None) -> dict:
    if not factor:
        slow = 0.8
        change_range = 0.4
        factor = slow + random.random() * change_range

    record["start"] = [value / factor for value in record["start"]]
    record["end"] = [value / factor for value in record["end"]]
    record["duration"] = [x - y for x, y in zip(record["end"], record["start"])]
    return record


def pitch_shift(pitch: list[int], shift_threshold: int = 5) -> list[int]:
    # No more than given number of steps
    PITCH_LOW = 21
    PITCH_HI = 108
    low_shift = -min(shift_threshold, min(pitch) - PITCH_LOW)
    high_shift = min(shift_threshold, PITCH_HI - max(pitch))

    if low_shift > high_shift:
        shift = 0
    else:
        shift = random.randint(low_shift, high_shift + 1)
    pitch = [value + shift for value in pitch]

    return pitch


def apply_augmentation(record: dict, quantizer: MidiQuantizer, augmentation_probability: float):
    augmented = record.copy()

    # check if augmentation happened
    done = False
    # shift pitch augmentation
    if random.random() < augmentation_probability:
        # max shift is octave down or up
        shift = random.randint(1, 12)
        augmented["pitch"] = pitch_shift(augmented["pitch"], shift)
        done = True

    # change tempo augmentation
    if random.random() < augmentation_probability:
        augmented = change_speed(augmented)

        # quantize dstart again
        dstart = []
        for it in range(len(augmented["start"]) - 1):
            dstart.append(augmented["start"][it + 1] - augmented["start"][it])
        dstart.append(0)

        augmented["dstart_bin"] = np.digitize(dstart, quantizer.dstart_bin_edges) - 1
        augmented["duration_bin"] = np.digitize(augmented["duration"], quantizer.duration_bin_edges) - 1
        done = True

    # if no augmentation was done, return None
    return augmented if done else None


def augment_dataset(
    dataset: Dataset,
    dataset_cfg: DictConfig,
    augmentation_probability: float = 0.5,
    augmentation_rep: int = 1,
):
    quantizer = MidiQuantizer(
        n_dstart_bins=dataset_cfg.quantization.dstart,
        n_duration_bins=dataset_cfg.quantization.duration,
        n_velocity_bins=dataset_cfg.quantization.velocity,
    )
    all_records = []
    for it, record in tqdm(enumerate(dataset), total=len(dataset)):
        augmented_records = [
            apply_augmentation(
                record=record,
                quantizer=quantizer,
                augmentation_probability=augmentation_probability,
            )
            for _ in range(augmentation_rep)
        ]
        records = [aug for aug in augmented_records if aug is not None] + [record]
        all_records += records
    augmented_dataset = Dataset.from_list(all_records)
    return augmented_dataset
