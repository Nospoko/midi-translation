from omegaconf import DictConfig
from training_utils import train_model
from data.dataset import load_cache_dataset
from data.tokenizer import QuantizedMidiEncoder, DstartEncoder


def main(cfg: DictConfig):
    src_encoder = QuantizedMidiEncoder(quantization_cfg=cfg.dataset.quantization)
    tgt_encoder = DstartEncoder(bins=cfg.dstart_bins)
    train_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name=cfg.dataset_name,
        split="train",
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )
    val_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name=cfg.dataset_name,
        split="validation",
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )

    train_model(train_dataset, val_dataset, cfg)

    print(cfg.run_name)
