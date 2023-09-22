from omegaconf import DictConfig

from training_utils import train_model
from data.tokenizer import DstartEncoder, QuantizedMidiEncoder
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset


def load_datasets(cfg: DictConfig) -> tuple[MyTokenizedMidiDataset, MyTokenizedMidiDataset]:
    src_encoder = QuantizedMidiEncoder(quantization_cfg=cfg.dataset.quantization)
    tgt_encoder = DstartEncoder(bins=cfg.dstart_bins)
    train_translation_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name=cfg.dataset_name,
        split="train",
    )
    val_translation_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name=cfg.dataset_name,
        split="validation",
    )

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
    return train_dataset, val_dataset


def main(cfg: DictConfig):
    train_dataset, val_dataset = load_datasets(cfg)

    train_model(train_dataset, val_dataset, cfg)

    print(cfg.run_name)
