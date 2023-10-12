import random
from os import cpu_count

import numpy as np
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
        # numbers from [low_shift, high_shift]
        shift = random.randint(low_shift, high_shift)
    pitch = [value + shift for value in pitch]

    return pitch


def apply_pitch_shift(batch: dict, augmentation_probability: float, shift: int):
    """
    Takes a batch of size 1 and applies augmentation - to use with dataset.map
    """
    assert len(batch["pitch"]) == 1
    if random.random() < augmentation_probability:
        augmented = batch.copy()
        # max shift is octave down or up
        augmented["pitch"][0] = pitch_shift(augmented["pitch"][0], shift)
        for key in batch.keys():
            batch[key].append(augmented[key][0])

    return batch


def apply_speed_change(batch: dict, quantizer: MidiQuantizer, augmentation_probability: float):
    """
    Takes a batch of size 1 and applies augmentation - to use with dataset.map
    """
    assert len(batch["pitch"]) == 1
    if random.random() < augmentation_probability:
        augmented = batch.copy()
        # I want to keep change_speed the same, so I have to unsqueeze the data
        time_vals = {"start": augmented["start"][0], "end": augmented["end"][0]}
        time_vals = change_speed(time_vals)

        augmented["start"][0] = time_vals["start"]
        augmented["end"][0] = time_vals["end"]
        augmented["duration"][0] = time_vals["duration"]

        # quantize dstart again
        dstart = []
        for it in range(len(augmented["start"][0]) - 1):
            dstart.append(augmented["start"][0][it + 1] - augmented["start"][0][it])
        dstart.append(0)

        augmented["dstart_bin"][0] = np.digitize(dstart, quantizer.dstart_bin_edges) - 1
        augmented["duration_bin"][0] = np.digitize(augmented["duration"][0], quantizer.duration_bin_edges) - 1
        for key in batch.keys():
            batch[key].append(augmented[key][0])

    return batch


def augment_dataset(dataset: Dataset, dataset_cfg: DictConfig, augmentation_cfg):
    """
    Augment the dataset with dataset.map method using all cpus.

    If augmentation_cfg.repetitions is 0, will output a copy of the dataset.
    """
    quantizer = MidiQuantizer(
        n_dstart_bins=dataset_cfg.quantization.dstart,
        n_duration_bins=dataset_cfg.quantization.duration,
        n_velocity_bins=dataset_cfg.quantization.velocity,
    )

    num_cpus = cpu_count()

    dataset = dataset.map(
        apply_pitch_shift,
        fn_kwargs={"shift": augmentation_cfg.shift, "augmentation_probability": augmentation_cfg.probability},
        batched=True,
        batch_size=1,
        num_proc=num_cpus,
    )
    dataset = dataset.map(
        apply_speed_change,
        fn_kwargs={"quantizer": quantizer, "augmentation_probability": augmentation_cfg.probability},
        batched=True,
        batch_size=1,
        num_proc=num_cpus,
    )
    return dataset
