import os

import numpy as np
import pandas as pd
import torch.nn as nn
import streamlit as st
from tqdm import tqdm
from fortepyan import MidiPiece
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from datasets import Dataset, load_dataset

from data.quantizer import MidiQuantizer
from utils import piece_av_files, decode_and_output
from data.dataset import MyTokenizedMidiDataset, quantized_piece_to_records
from data.tokenizer import MidiEncoder, VelocityEncoder, QuantizedMidiEncoder


def creative_prompts(model: nn.Module, train_cfg: DictConfig, model_dir: str):
    quantizer = MidiQuantizer(
        n_dstart_bins=train_cfg.dataset.quantization.dstart,
        n_duration_bins=train_cfg.dataset.quantization.duration,
        n_velocity_bins=train_cfg.dataset.quantization.velocity,
    )
    st.markdown(f"Velocity bins: {quantizer.velocity_bin_edges}")
    src_encoder = QuantizedMidiEncoder(train_cfg.dataset.quantization)
    tgt_encoder = VelocityEncoder()

    dataset_name = st.text_input(label="dataset", value=train_cfg.dataset_name)
    split = st.text_input(label="split", value="test")
    record_id = st.number_input(label="record id", value=0)
    hf_dataset = load_dataset(dataset_name, split=split)

    # Select one full piece
    record = hf_dataset[record_id]
    piece = MidiPiece.from_huggingface(record)
    st.markdown(f"Piece size: {piece.size}")

    start_note = st.number_input(label="first note index", value=0)

    segments_to_process = 2
    notes_to_process = segments_to_process * train_cfg.dataset.sequence_len
    finish = start_note + notes_to_process
    gt_piece = piece[start_note:finish]

    save_base_pred = f"{dataset_name}-{split}-{record_id}-{start_note}-{train_cfg.run_name}".replace("/", "_")
    save_base_pred = os.path.join(model_dir, save_base_pred)
    gt_paths = piece_av_files(gt_piece, save_base=save_base_pred)

    st.markdown("### Original")
    st.json(gt_piece.source)
    st.image(gt_paths["pianoroll_path"])
    st.audio(gt_paths["mp3_path"])

    # Two sines every other note
    v3_piece = piece[start_note:finish]
    v3_prompt = two_sines_prompt(v3_piece)
    v3_piece.df.velocity = v3_prompt

    v3_velocity = generate_velocities(
        model=model,
        piece=v3_piece,
        train_cfg=train_cfg,
        quantizer=quantizer,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )
    v3_piece.df.velocity = v3_velocity

    save_base_v3 = save_base_pred + "-v3"
    v3_paths = piece_av_files(v3_piece, save_base=save_base_v3)

    v3_fig = velocity_comparison_figure(
        gt_piece=gt_piece,
        velocity_prompt=v3_prompt,
        quantized_prompt=quantizer.quantize_velocity(v3_prompt),
        generated_velocity=v3_velocity.values,
    )
    st.markdown("### Two sines every other note")
    st.pyplot(v3_fig)
    st.image(v3_paths["pianoroll_path"])
    st.audio(v3_paths["mp3_path"])

    # Low notes sine
    v4_piece = piece[start_note:finish]
    v4_prompt = low_sine_prompt(v4_piece)
    v4_piece.df.velocity = v4_prompt

    v4_velocity = generate_velocities(
        model=model,
        piece=v4_piece,
        train_cfg=train_cfg,
        quantizer=quantizer,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )
    v4_piece.df.velocity = v4_velocity

    save_base_v4 = save_base_pred + "-v4"
    v4_paths = piece_av_files(v4_piece, save_base=save_base_v4)

    v4_fig = velocity_comparison_figure(
        gt_piece=gt_piece,
        velocity_prompt=v4_prompt,
        quantized_prompt=quantizer.quantize_velocity(v4_prompt),
        generated_velocity=v4_velocity.values,
    )
    st.markdown("### Sine in low notes only")
    st.pyplot(v4_fig)
    st.image(v4_paths["pianoroll_path"])
    st.audio(v4_paths["mp3_path"])

    # constant velocities
    v5_piece = piece[start_note:finish]
    v5_prompt = 70 * np.ones_like(v5_piece.df.velocity)
    v5_piece.df.velocity = v5_prompt
    v5_velocity = generate_velocities(
        model=model,
        piece=v5_piece,
        train_cfg=train_cfg,
        quantizer=quantizer,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )
    v5_piece.df.velocity = v5_velocity

    save_base_v5 = save_base_pred + "-v5"
    v5_paths = piece_av_files(v5_piece, save_base=save_base_v5)

    v5_fig = velocity_comparison_figure(
        gt_piece=gt_piece,
        velocity_prompt=v5_prompt,
        quantized_prompt=quantizer.quantize_velocity(v5_prompt),
        generated_velocity=v5_velocity.values,
    )

    st.markdown("### Constant initialization")
    st.pyplot(v5_fig)
    st.image(v5_paths["pianoroll_path"])
    st.audio(v5_paths["mp3_path"])

    # Random velocities
    v1_piece = piece[start_note:finish]
    v1_prompt = np.random.randint(128, size=v1_piece.size)
    v1_piece.df.velocity = v1_prompt
    v1_velocity = generate_velocities(
        model=model,
        piece=v1_piece,
        train_cfg=train_cfg,
        quantizer=quantizer,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )
    v1_piece.df.velocity = v1_velocity

    save_base_v1 = save_base_pred + "-v1"
    v1_paths = piece_av_files(v1_piece, save_base=save_base_v1)

    v1_fig = velocity_comparison_figure(
        gt_piece=gt_piece,
        velocity_prompt=v1_prompt,
        quantized_prompt=quantizer.quantize_velocity(v1_prompt),
        generated_velocity=v1_velocity.values,
    )

    st.markdown("### Random initialization")
    st.pyplot(v1_fig)
    st.image(v1_paths["pianoroll_path"])
    st.audio(v1_paths["mp3_path"])


def two_sines_prompt(piece: MidiPiece) -> np.array:
    n_left = piece.size // 2
    x_left = np.linspace(0, 10, n_left)
    y_left = 70 + 30 * np.sin(x_left)

    n_right = piece.size - n_left
    x_right = np.linspace(0, 10, n_right)
    y_right = 70 - 30 * np.sin(x_right)
    prompt_velocities = np.column_stack([y_left, y_right]).ravel().astype(int)

    return prompt_velocities


def low_sine_prompt(piece: MidiPiece) -> np.array:
    df = piece.df

    # Take the half of the notes on the lower side
    median_pitch = df.pitch.median()
    low_ids = df.pitch < median_pitch

    # And make them velocity sine
    x_low = np.linspace(0, 10, low_ids.sum())
    y_low = 70 + 30 * np.sin(x_low)
    low_prompt = y_low.astype(int)

    # Make a copy of the ground truth values
    prompt_velocities = df.velocity.values
    prompt_velocities[low_ids] = low_prompt
    return prompt_velocities


def generate_velocities(
    model: nn.Module,
    piece: MidiPiece,
    train_cfg: DictConfig,
    quantizer: MidiQuantizer,
    src_encoder: MidiEncoder,
    tgt_encoder: MidiEncoder,
) -> pd.DataFrame:
    # And run full pre-processing ...
    qpiece = quantizer.inject_quantization_features(piece)
    sequences = quantized_piece_to_records(
        piece=qpiece,
        sequence_len=train_cfg.dataset.sequence_len,
        sequence_step=train_cfg.dataset.sequence_len,
    )
    one_record_dataset = Dataset.from_list(sequences)

    # ... to get it into a format the model understands
    dataset = MyTokenizedMidiDataset(
        dataset=one_record_dataset,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
        dataset_cfg=train_cfg.dataset,
    )

    pad_idx = src_encoder.token_to_id["<blank>"]

    predicted_tokens = []
    for record in tqdm(dataset):
        src_token_ids = record["source_token_ids"]
        src_mask = (src_token_ids != pad_idx).unsqueeze(-2)

        predicted_token_ids, probabilities = decode_and_output(
            model=model,
            src=src_token_ids,
            src_mask=src_mask[0],
            max_len=train_cfg.dataset.sequence_len,
            start_symbol=0,
            device=train_cfg.device,
        )

        out_tokens = [tgt_encoder.vocab[x] for x in predicted_token_ids if x != pad_idx]
        predicted_tokens += out_tokens

    pred_velocities = tgt_encoder.untokenize(predicted_tokens)

    return pred_velocities


def velocity_comparison_figure(
    gt_piece: MidiPiece,
    velocity_prompt: np.array,
    quantized_prompt: np.array,
    generated_velocity: np.array,
) -> plt.Figure:
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=[8, 3],
        gridspec_kw={
            "hspace": 0,
        },
    )
    ax = axes[0]
    df = gt_piece.df
    ax.plot(df.start, df.velocity, "o", ms=7, label="truth")
    ax.plot(df.start, df.velocity, ".", color="white")
    ax.vlines(
        df.start,
        ymin=0,
        ymax=df.velocity,
        lw=2,
        alpha=0.777,
    )
    ax.set_ylim(0, 160)
    # Add a grid to the plot
    ax.grid()
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(df.start, velocity_prompt, "o", ms=7, label="prompt")
    ax.plot(df.start, velocity_prompt, ".", color="white")
    ax.vlines(
        df.start,
        ymin=0,
        ymax=velocity_prompt,
        lw=2,
        alpha=0.777,
    )
    ax.set_ylim(0, 160)
    # Add a grid to the plot
    ax.grid()
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(df.start, quantized_prompt, "o", ms=7, label="quantized")
    ax.plot(df.start, quantized_prompt, ".", color="white")
    ax.vlines(
        df.start,
        ymin=0,
        ymax=quantized_prompt,
        lw=2,
        alpha=0.777,
    )
    ax.set_ylim(0, 160)
    # Add a grid to the plot
    ax.grid()
    ax.legend(loc="upper right")

    ax = axes[3]
    ax.plot(df.start, generated_velocity, "o", ms=7, label="generated")
    ax.plot(df.start, generated_velocity, ".", color="white")
    ax.vlines(
        df.start,
        ymin=0,
        ymax=generated_velocity,
        lw=2,
        alpha=0.777,
    )
    ax.set_ylim(0, 160)
    # Add a grid to the plot
    ax.grid()
    ax.legend(loc="upper right")

    return fig
