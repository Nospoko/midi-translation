from omegaconf import DictConfig

from training_utils import train_model
from data.augmentation import augment_dataset
from data.tokenizer import DstartEncoder, QuantizedMidiEncoder
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset


def load_datasets(cfg: DictConfig) -> tuple[MyTokenizedMidiDataset, MyTokenizedMidiDataset]:
    src_encoder = QuantizedMidiEncoder(quantization_cfg=cfg.dataset.quantization)
    tgt_encoder = DstartEncoder(n_bins=cfg.dstart_bins)

    if "maestro" in cfg.dataset_name:
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
    else:
        translation_dataset = load_cache_dataset(dataset_cfg=cfg.dataset, dataset_name=cfg.dataset_name, split="train")
        train_translation_dataset, val_translation_dataset = translation_dataset.train_test_split(0.1).values()

    # TODO: move rep and probability to config (?)
    train_translation_dataset = augment_dataset(
        dataset=train_translation_dataset,
        dataset_cfg=cfg.dataset,
        augmentation_probability=0.5,
        augmentation_rep=1,
    )
    val_translation_dataset = augment_dataset(
        dataset=val_translation_dataset,
        dataset_cfg=cfg.dataset,
        augmentation_probability=0.5,
        augmentation_rep=1,
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
