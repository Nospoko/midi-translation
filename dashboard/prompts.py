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
from dashboard.components import download_button
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
    v3_prompt = two_sines_prompt(gt_piece)
    render_prompt_results(
        save_base=save_base_pred,
        model=model,
        prompt=v3_prompt,
        prompt_title="Two Sines",
        gt_piece=gt_piece,
        train_cfg=train_cfg,
        quantizer=quantizer,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )

    # Low notes sine
    v4_prompt = low_sine_prompt(gt_piece)
    render_prompt_results(
        save_base=save_base_pred,
        model=model,
        prompt=v4_prompt,
        prompt_title="Low Notes Sine",
        gt_piece=gt_piece,
        train_cfg=train_cfg,
        quantizer=quantizer,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )

    # constant velocities
    v5_prompt = 70 * np.ones_like(gt_piece.df.velocity)
    render_prompt_results(
        save_base=save_base_pred,
        model=model,
        prompt=v5_prompt,
        prompt_title="Constant Initialization",
        gt_piece=gt_piece,
        train_cfg=train_cfg,
        quantizer=quantizer,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )

    # Random velocities
    v1_prompt = np.random.randint(128, size=gt_piece.size)
    render_prompt_results(
        save_base=save_base_pred,
        model=model,
        prompt=v1_prompt,
        prompt_title="Random Initialization",
        gt_piece=gt_piece,
        train_cfg=train_cfg,
        quantizer=quantizer,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )


def render_prompt_results(
    save_base: str,
    model: nn.Module,
    prompt: np.array,
    prompt_title: str,
    gt_piece: MidiPiece,
    train_cfg: DictConfig,
    quantizer: MidiQuantizer,
    src_encoder: MidiEncoder,
    tgt_encoder: MidiEncoder,
):
    # This copies a fortepyan piece
    piece = gt_piece[:]
    piece.df.velocity = prompt
    v1_velocity = generate_velocities(
        model=model,
        piece=piece,
        train_cfg=train_cfg,
        quantizer=quantizer,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )
    piece.df.velocity = v1_velocity

    prompt_save_base = save_base + prompt_title.lower().replace(" ", "_")
    paths = piece_av_files(piece, save_base=prompt_save_base)

    fig = velocity_comparison_figure(
        gt_piece=gt_piece,
        velocity_prompt=prompt,
        quantized_prompt=quantizer.quantize_velocity(prompt),
        generated_velocity=v1_velocity.values,
    )

    st.markdown(f"### {prompt_title}")
    st.pyplot(fig)
    st.image(paths["pianoroll_path"])
    st.audio(paths["mp3_path"])

    midi_path = paths["midi_path"]
    with open(midi_path, "rb") as file:
        download_button_str = download_button(
            object_to_download=file.read(),
            download_filename=midi_path.split("/")[-1],
            button_text="Download midi",
        )
        st.markdown(download_button_str, unsafe_allow_html=True)


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
    prompt_velocities = df.velocity.values.copy()
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
    # Helper drawer
    def draw_velocity(ax: plt.Axes, start: np.array, velocity: np.array, label: str):
        ax.plot(start, velocity, "o", ms=7, label=label)
        ax.plot(start, velocity, ".", color="white")
        ax.vlines(
            start,
            ymin=0,
            ymax=velocity,
            lw=2,
            alpha=0.777,
        )
        ax.set_ylim(0, 160)
        # Add a grid to the plot
        ax.grid()
        ax.legend(loc="upper right")

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=[8, 3],
        gridspec_kw={
            "hspace": 0,
        },
    )

    df = gt_piece.df

    velocities = [df.velocity, velocity_prompt, quantized_prompt, generated_velocity]
    labels = ["truth", "prompt", "quantized", "generated"]
    for it, ax in enumerate(axes):
        draw_velocity(
            ax=ax,
            start=df.start,
            velocity=velocities[it],
            label=labels[it],
        )

    return fig
