from datasets import Dataset
from omegaconf import DictConfig

from training_utils import train_model
from data.dataset import MyTokenizedMidiDataset
from data.tokenizer import DstartEncoder, QuantizedMidiEncoder


def main(cfg: DictConfig, train_translation_dataset: Dataset, val_translation_dataset: Dataset):
    src_encoder = QuantizedMidiEncoder(quantization_cfg=cfg.dataset.quantization)
    tgt_encoder = DstartEncoder(n_bins=cfg.dstart_bins)

    train_dataset = MyTokenizedMidiDataset(
        dataset=train_translation_dataset,
        dataset_cfg=cfg.dataset,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )
    val_dataset = MyTokenizedMidiDataset(
        dataset=val_translation_dataset,
        dataset_cfg=cfg.dataset,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )

    train_model(train_dataset, val_dataset, cfg)

    print(cfg.run_name)
