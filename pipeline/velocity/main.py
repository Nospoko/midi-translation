from omegaconf import DictConfig
from datasets import concatenate_datasets

from training_utils import train_model
from data.augmentation import augment_dataset
from data.tokenizer import VelocityEncoder, QuantizedMidiEncoder
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset


def load_train_dataset(cfg: DictConfig):
    datasets = []
    for name in cfg.dataset_name.split("+"):
        dataset = load_cache_dataset(
            dataset_cfg=cfg.dataset,
            dataset_name=name,
            split="train",
        )
        datasets.append(dataset)
    train_dataset = concatenate_datasets(datasets)

    return train_dataset


def load_datasets(cfg: DictConfig) -> tuple[MyTokenizedMidiDataset, MyTokenizedMidiDataset]:
    src_encoder = QuantizedMidiEncoder(quantization_cfg=cfg.dataset.quantization)
    tgt_encoder = VelocityEncoder()

    train_translation_dataset = load_train_dataset(cfg)
    val_translation_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name="roszcz/maestro-v1-sustain",
        split="test+validation",
    )

    if cfg.augmentation.repetitions > 0:
        train_translation_dataset = augment_dataset(
            dataset=train_translation_dataset,
            dataset_cfg=cfg.dataset,
            augmentation_cfg=cfg.augmentation,
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
