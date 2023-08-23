import pandas as pd
from datasets import load_dataset


class MidiTranslationDataset:
    def __init__(
            self,
            split: str = "train",
    ):
        self.dataset = load_dataset("roszcz/maestro-v1", split=split)
        self.samples = []