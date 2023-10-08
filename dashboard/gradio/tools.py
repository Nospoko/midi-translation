from __future__ import annotations

import shutil
import tempfile
import subprocess

import PIL
import torch
import PIL.Image
import torch.nn as nn
import fortepyan as ff
from tqdm import tqdm
import matplotlib.pyplot as plt
from gradio import utils, processing_utils
from omegaconf import OmegaConf, DictConfig
from fortepyan.audio import render as render_audio

from model import make_model
from data.quantizer import MidiQuantizer
from utils import vocab_sizes, greedy_decode
from data.dataset import quantized_piece_to_records
from data.tokenizer import DstartEncoder, VelocityEncoder, QuantizedMidiEncoder


def quantize_piece(train_cfg: DictConfig, piece: ff.MidiPiece) -> ff.MidiPiece:
    quantizer = MidiQuantizer(
        n_dstart_bins=train_cfg.dataset.quantization.dstart,
        n_duration_bins=train_cfg.dataset.quantization.duration,
        n_velocity_bins=train_cfg.dataset.quantization.velocity,
    )

    # pre-process the piece ...
    qpiece = quantizer.inject_quantization_features(piece)

    qpiece.df = quantizer.apply_quantization(qpiece.df)
    return ff.MidiPiece(qpiece.df)


def process_piece(train_cfg: DictConfig, piece: ff.MidiPiece) -> list[torch.Tensor]:
    """
    Run full pre-processing on a piece
    """
    src_encoder = QuantizedMidiEncoder(train_cfg.dataset.quantization)
    quantizer = MidiQuantizer(
        n_dstart_bins=train_cfg.dataset.quantization.dstart,
        n_duration_bins=train_cfg.dataset.quantization.duration,
        n_velocity_bins=train_cfg.dataset.quantization.velocity,
    )

    # pre-process the piece ...
    qpiece = quantizer.inject_quantization_features(piece)
    sequences = quantized_piece_to_records(
        piece=qpiece,
        sequence_len=train_cfg.dataset.sequence_len,
        sequence_step=train_cfg.dataset.sequence_len,
    )

    sequence_tokens = [
        torch.tensor([src_encoder.token_to_id["<CLS>"]] + src_encoder.encode(sequence), dtype=torch.int64)
        for sequence in sequences
    ]

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


def predict_dstart(model: nn.Module, train_cfg: DictConfig, piece: ff.MidiPiece) -> ff.MidiPiece:
    tgt_encoder = DstartEncoder(n_bins=train_cfg.dstart_bins)
    sequences = process_piece(train_cfg=train_cfg, piece=piece)

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

    predictions = tgt_encoder.untokenize(predicted_tokens)

    predicted_piece_df = piece.df.copy()
    predicted_piece_df = predicted_piece_df.head(len(predictions))

    predicted_piece_df["start"] = tgt_encoder.unquantized_start(predictions)
    predicted_piece_df["end"] = predicted_piece_df["start"] + predicted_piece_df["duration"]
    predicted_piece = ff.MidiPiece(predicted_piece_df)

    predicted_piece.source = piece.source.copy()

    return predicted_piece


def predict_velocity(model: nn.Module, train_cfg: DictConfig, piece: ff.MidiPiece):
    """
    Returns a MidiPiece with velocity predicted by the model.
    """
    tgt_encoder = VelocityEncoder()
    sequences = process_piece(train_cfg=train_cfg, piece=piece)

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


def make_pianoroll_video(
    piece: ff.MidiPiece,
) -> str:
    audio_file = render_audio.midi_to_mp3(piece.to_midi(), "tmp/predicted-audio.mp3")
    audio = processing_utils.audio_from_file(audio_file)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found.")

    duration = round(len(audio[1]) / audio[0], 4)

    with utils.MatplotlibBackendMananger():
        ff.view.draw_pianoroll_with_velocities(piece)
        plt.tight_layout()

        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        savefig_kwargs = {"bbox_inches": "tight"}
        plt.savefig(tmp_img.name, **savefig_kwargs)
        plt.clf()

        waveform_img = PIL.Image.open(tmp_img.name)

        img_width, img_height = waveform_img.size
        waveform_img.save(tmp_img.name)

    # Convert waveform to video with ffmpeg
    output_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ffmpeg_cmd = [
        ffmpeg,
        "-loop",
        "1",
        "-i",
        tmp_img.name,
        "-i",
        audio_file,
        "-vf",
        f"color=c=#FFFFFF77:s={img_width}x{img_height}[bar];[0][bar]overlay=-w+(w/{duration})*t:H-h:shortest=1",
        "-t",
        str(duration),
        "-y",
        output_mp4.name,
    ]

    subprocess.check_call(ffmpeg_cmd)
    return output_mp4.name
