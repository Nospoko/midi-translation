from data.dataset import MyTokenizedMidiDataset
from data.tokenizer import QuantizedMidiEncoder, DstartEncoder
from omegaconf import DictConfig
import torch.nn as nn
from datasets import Dataset
from utils import decode_and_output, calculate_average_distance
from modules.label_smoothing import LabelSmoothing
from utils import vocab_sizes
from tqdm import tqdm


def main(cfg: DictConfig, model: nn.Module, translation_dataset: Dataset, device: str = "cpu"):
    src_encoder = QuantizedMidiEncoder(quantization_cfg=cfg.dataset.quantization)
    tgt_encoder = DstartEncoder(bins=cfg.dstart_bins)

    dataset = MyTokenizedMidiDataset(
        dataset=translation_dataset,
        dataset_cfg=cfg.dataset,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )

    pad_idx = src_encoder.token_to_id["<blank>"]
    start_symbol = src_encoder.token_to_id["<s>"]
    total_loss = 0
    total_dist = 0
    _, tgt_vocab_size = vocab_sizes(cfg)
    criterion = LabelSmoothing(
        size=tgt_vocab_size,
        padding_idx=pad_idx,
        smoothing=cfg.train.label_smoothing,
    )
    criterion.to(device)
    pbar = tqdm(dataset)
    counter = 0
    for record in pbar:
        counter += 1
        src = record["source_token_ids"]
        tgt = record["target_token_ids"]

        src_mask = (src != pad_idx).unsqueeze(-2)

        _, out = decode_and_output(
            model=model,
            src=src,
            src_mask=src_mask,
            max_len=cfg.dataset.sequence_len,
            start_symbol=start_symbol,
            device=device
        )

        target = tgt[1:-1].to(device)
        n_tokens = (target != pad_idx).data.sum()

        loss = criterion(out, target) / n_tokens
        total_loss += loss.item()
        total_dist += calculate_average_distance(out, target).cpu()

        pbar.set_description(f"average distance: {total_dist / counter:3.3}, average loss: {total_loss / counter:3.3}")

    avg_loss = total_loss / len(dataset)
    avg_dist = total_dist / len(dataset)
    print(f"Average loss: {avg_loss}")
    print(f"Average distance: {avg_dist}")
