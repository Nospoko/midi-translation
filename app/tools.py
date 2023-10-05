import torch
import torch.nn as nn
import fortepyan as ff
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from model import make_model
from data.quantizer import MidiQuantizer
from utils import vocab_sizes, greedy_decode
from data.dataset import quantized_piece_to_records
from data.tokenizer import VelocityEncoder, QuantizedMidiEncoder


def encode_record(sequence: dict, encoder: QuantizedMidiEncoder) -> torch.Tensor:
    token_ids = encoder.encode(sequence)
    token_ids = [encoder.token_to_id["<CLS>"]] + token_ids
    return torch.tensor(token_ids, dtype=torch.int64)


def process_piece(train_cfg: DictConfig, piece: ff.MidiPiece, quantizer: MidiQuantizer) -> list[torch.Tensor]:
    """
    Run full pre-processing on a piece
    """
    src_encoder = QuantizedMidiEncoder(train_cfg.dataset.quantization)

    # pre-process the piece ...
    qpiece = quantizer.inject_quantization_features(piece)
    sequences = quantized_piece_to_records(
        piece=qpiece,
        sequence_len=train_cfg.dataset.sequence_len,
        sequence_step=train_cfg.dataset.sequence_len,
    )
    sequence_tokens = [encode_record(sequence, src_encoder) for sequence in sequences]

    return sequence_tokens


def load_model(checkpoint: dict) -> nn.Module:
    cfg = OmegaConf.create(checkpoint["cfg"])
    src_vocab_size, tgt_vocab_size = vocab_sizes(cfg)
    model = make_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        n=cfg.model.n,
        h=cfg.model.h,
        d_ff=cfg.model.d_ff,
        d_model=cfg.model.d_model,
    )
    model.to("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def predict_velocity(model: nn.Module, train_cfg: DictConfig, piece: ff.MidiPiece):
    """
    Returns a MidiPiece with velocity predicted by the model.
    """
    tgt_encoder = VelocityEncoder()
    quantizer = MidiQuantizer(
        n_dstart_bins=train_cfg.dataset.quantization.dstart,
        n_duration_bins=train_cfg.dataset.quantization.duration,
        n_velocity_bins=train_cfg.dataset.quantization.velocity,
    )
    sequences = process_piece(train_cfg=train_cfg, piece=piece, quantizer=quantizer)

    predicted_tokens = []
    for record in tqdm(sequences):
        src_token_ids = record

        predicted_token_ids = greedy_decode(
            model=model,
            src=src_token_ids,
            max_len=train_cfg.dataset.sequence_len,
        )

        out_tokens = [tgt_encoder.vocab[x] for x in predicted_token_ids]
        predicted_tokens += out_tokens

    pred_velocities = tgt_encoder.untokenize(predicted_tokens)

    predicted_piece_df = piece.df.copy()
    predicted_piece_df = predicted_piece_df.head(len(pred_velocities))

    predicted_piece_df["velocity"] = pred_velocities
    predicted_piece = ff.MidiPiece(predicted_piece_df)

    predicted_piece.source = piece.source.copy()
    return predicted_piece
