import os

import numpy as np
import pandas as pd
import torch.nn as nn
import streamlit as st
from tqdm import tqdm
from fortepyan import MidiPiece
from omegaconf import DictConfig
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

    segments_to_process = 2
    notes_to_process = segments_to_process * train_cfg.dataset.sequence_len
    gt_piece = piece[:notes_to_process]

    save_base_pred = f"{dataset_name}-{split}-{record_id}-{train_cfg.run_name}".replace("/", "_")
    save_base_pred = os.path.join(model_dir, save_base_pred)
    gt_paths = piece_av_files(gt_piece, save_base=save_base_pred)

    st.markdown("### Original")
    st.json(gt_piece.source)
    st.image(gt_paths["pianoroll_path"])
    st.audio(gt_paths["mp3_path"])

    # Two sines every other note
    v3_piece = piece[:notes_to_process]
    n_left = v3_piece.size // 2
    x_left = np.linspace(0, 10, n_left)
    y_left = 70 + 30 * np.sin(x_left)

    n_right = v3_piece.size - n_left
    x_right = np.linspace(0, 10, n_right)
    y_right = 70 - 30 * np.sin(x_right)
    v3_velocity = np.column_stack([y_left, y_right]).ravel()
    v3_piece.df.velocity = v3_velocity.astype(int)

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

    st.markdown("### Two sines every other note")
    st.image(v3_paths["pianoroll_path"])
    st.audio(v3_paths["mp3_path"])

    # Random velocities
    v1_piece = piece[:notes_to_process]
    v1_piece.df.velocity = np.random.randint(128, size=v1_piece.size)
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

    st.markdown("### Random initialization")
    st.image(v1_paths["pianoroll_path"])
    st.audio(v1_paths["mp3_path"])


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
