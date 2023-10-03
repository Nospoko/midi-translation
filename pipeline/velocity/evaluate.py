import torch.nn as nn
from tqdm import tqdm
from datasets import Dataset
from omegaconf import DictConfig

from data.dataset import MyTokenizedMidiDataset
from modules.label_smoothing import LabelSmoothing
from data.tokenizer import VelocityEncoder, QuantizedMidiEncoder
from utils import vocab_sizes, decode_and_output, calculate_average_distance


def main(cfg: DictConfig, model: nn.Module, translation_dataset: Dataset, device: str = "cpu"):
    src_encoder = QuantizedMidiEncoder(quantization_cfg=cfg.dataset.quantization)
    tgt_encoder = VelocityEncoder()

    dataset = MyTokenizedMidiDataset(
        dataset=translation_dataset,
        dataset_cfg=cfg.dataset,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )

    total_loss = 0
    total_dist = 0
    _, tgt_vocab_size = vocab_sizes(cfg)
    criterion = LabelSmoothing(
        size=tgt_vocab_size,
        smoothing=cfg.train.label_smoothing,
    )
    criterion.to(device)
    pbar = tqdm(dataset)
    counter = 0
    for record in pbar:
        counter += 1

        src = record["source_token_ids"]
        tgt = record["target_token_ids"]

        _, out = decode_and_output(
            model=model,
            src=src,
            max_len=cfg.dataset.sequence_len,
            device=device,
        )

        target = tgt[1:].to(device)
        n_tokens = target.numel()

        loss = criterion(out, target) / n_tokens
        total_loss += loss.item()
        total_dist += calculate_average_distance(out, target).cpu()

        desc = f"average distance: {total_dist / counter:3.3f}, average loss: {total_loss / counter:3.3f}"
        pbar.set_description(desc)

    avg_loss = total_loss / len(dataset)
    avg_dist = total_dist / len(dataset)
    print(f"Average loss: {avg_loss}")
    print(f"Average distance: {avg_dist}")
