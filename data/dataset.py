import itertools

import fortepyan as ff
from tqdm import tqdm
from datasets import load_dataset
from prepare_dataset import process_record

from data.quantizer import MidiQuantizer


class TokenizedMidiTranslationDataset:
    def __init__(
        self,
        split: str = "train",
        n_dstart_bins: int = 3,
        n_duration_bins: int = 10,
        n_velocity_bins: int = 10,
        sequence_len: int = 128,
    ):
        self.sequence_len = sequence_len
        self.quantizer = MidiQuantizer(
            n_dstart_bins=n_dstart_bins,
            n_duration_bins=n_duration_bins,
            n_velocity_bins=n_velocity_bins,
        )
        self.dataset = load_dataset(path="roszcz/maestro-v1", split=split)
        self.processed_records = []
        self.samples = []
        self.vocab_src, self.vocab_tgt = self._build_vocab()
        self._build()

    def _build(self):
        for record in tqdm(self.dataset, total=self.dataset.num_rows):
            piece = ff.MidiPiece.from_huggingface(record)
            processed_record = process_record(piece, self.sequence_len, self.quantizer)

            self.processed_records += processed_record

    def _build_vocab(self) -> tuple[list[str], list[str]]:
        vocab_src, vocab_tgt = [], []

        # every combination of pitch + dstart
        product = itertools.product(range(21, 109), range(self.quantizer.n_dstart_bins))
        for pitch, dstart in product:
            key = f"{pitch}-{dstart}"
            vocab_src.append(key)

        # every combination of duration + velocity
        product = itertools.product(range(self.quantizer.n_duration_bins), range(self.quantizer.n_velocity_bins))
        for duration, velocity in product:
            key = f"{duration}-{velocity}"
            vocab_tgt.append(key)
        return vocab_src, vocab_tgt

    def __getitem__(self, idx: int):
        return self.processed_records[idx]


if __name__ == "__main__":
    dataset = TokenizedMidiTranslationDataset()
    print(dataset[0])
