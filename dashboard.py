import os
import glob
import json

import torch
import numpy as np
import pandas as pd
import streamlit as st
from fortepyan import MidiPiece
from omegaconf import OmegaConf
from datasets import load_dataset

from model import make_model
from data.quantizer import MidiQuantizer
from data.dataset import load_cache_dataset
from predict_piece import predict_piece_dashboard
from utils import vocab_sizes, piece_av_files, generate_sequence

# For now let's run all dashboards on CPU
DEVICE = "cpu"


def main():
    with st.sidebar:
        mode = st.selectbox(label="Display", options=["Model predictions", "Predict piece", "Tokenization review"])

    if mode == "Tokenization review":
        tokenization_review_dashboard()
    if mode == "Predict piece":
        predict_piece_dashboard()
    if mode == "Model predictions":
        model_predictions_review()


def model_predictions_review():
    with st.sidebar:
        # options
        path = st.selectbox(label="model", options=glob.glob("models/*.pt"))
        st.markdown("Selected checkpoint:")
        st.markdown(path)

    # load checkpoint, force dashboard device
    checkpoint = torch.load(path, map_location=DEVICE)
    train_cfg = OmegaConf.create(checkpoint["cfg"])
    train_cfg.device = DEVICE

    st.markdown("Model config:")
    model_params = OmegaConf.to_container(train_cfg.model)
    st.json(model_params, expanded=False)

    dataset_params = OmegaConf.to_container(train_cfg.dataset)
    st.markdown("Dataset config:")
    st.json(dataset_params, expanded=True)

    dataset_cfg = train_cfg.dataset
    dataset_name = st.text_input(label="dataset", value=train_cfg.dataset_name)
    split = st.text_input(label="split", value="test")

    # Prepare everything required to make inference
    quantizer = MidiQuantizer(
        n_dstart_bins=dataset_cfg.quantization.dstart,
        n_duration_bins=dataset_cfg.quantization.duration,
        n_velocity_bins=dataset_cfg.quantization.velocity,
    )

    random_seed = st.selectbox(label="random seed", options=range(20))

    dataset = load_cache_dataset(
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        split=split,
    )
    src_vocab_size, tgt_vocab_size = vocab_sizes(train_cfg)

    model = make_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        n=train_cfg.model.n,
        d_model=train_cfg.model.d_model,
        d_ff=train_cfg.model.d_ff,
        h=train_cfg.model.h,
        dropout=train_cfg.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    n_samples = 5
    np.random.seed(random_seed)
    idxs = np.random.randint(len(dataset), size=n_samples)

    pad_idx = dataset.src_encoder.token_to_id["<blank>"]

    cols = st.columns(5)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Quantized")
    with cols[2]:
        st.markdown("### Q. velocity")
    with cols[3]:
        st.markdown("### Predicted")

    # predict velocities and get src, tgt and model output
    print("Making predictions ...")
    for record_id in idxs:
        # Numpy to int :(
        record = dataset.get_complete_record(int(record_id))
        record_source = json.loads(record["source"])
        src_token_ids = record["source_token_ids"]

        generated_velocity = generate_sequence(
            model=model,
            pad_idx=pad_idx,
            src_tokens=src_token_ids,
            tgt_encoder=dataset.tgt_encoder,
            sequence_size=train_cfg.dataset.sequence_len,
        )

        # Just pitches and quantization bins of the source
        src_tokens = [dataset.src_encoder.vocab[token_id] for token_id in src_token_ids if token_id != pad_idx]
        source_df = dataset.src_encoder.untokenize(src_tokens)

        quantized_notes = quantizer.apply_quantization(source_df)
        quantized_piece = MidiPiece(quantized_notes)
        quantized_piece.time_shift(-quantized_piece.df.start.min())

        # TODO start here
        # Reconstruct the sequence as recorded
        midi_columns = ["pitch", "start", "end", "duration", "velocity"]
        true_notes = pd.DataFrame({c: record[c] for c in midi_columns})
        true_piece = MidiPiece(df=true_notes, source=record_source)
        true_piece.time_shift(-true_piece.df.start.min())

        pred_piece_df = true_piece.df.copy()
        quantized_vel_df = true_piece.df.copy()

        # change untokenized velocities to model predictions
        pred_piece_df["velocity"] = generated_velocity
        pred_piece_df["velocity"] = pred_piece_df["velocity"].fillna(0)

        quantized_vel_df["velocity"] = quantized_piece.df["velocity"].copy()

        # create quantized piece with predicted velocities
        pred_piece = MidiPiece(pred_piece_df)
        quantized_vel_piece = MidiPiece(quantized_vel_df)

        pred_piece.source = true_piece.source.copy()
        quantized_vel_piece.source = true_piece.source.copy()

        model_dir = f"tmp/dashboard/{train_cfg.run_name}"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # create files
        true_save_base = os.path.join(model_dir, f"true_{record_id}")
        true_piece_paths = piece_av_files(piece=true_piece, save_base=true_save_base)

        src_save_base = os.path.join(model_dir, f"src_{record_id}")
        src_piece_paths = piece_av_files(piece=quantized_piece, save_base=src_save_base)

        qv_save_base = os.path.join(model_dir, f"qv_{record_id}")
        qv_paths = piece_av_files(piece=quantized_vel_piece, save_base=qv_save_base)

        predicted_save_base = os.path.join(model_dir, f"predicted_{record_id}")
        predicted_paths = piece_av_files(piece=pred_piece, save_base=predicted_save_base)

        # create a dashboard
        st.json(record_source)
        cols = st.columns(4)
        with cols[0]:
            # Unchanged
            st.image(true_piece_paths["pianoroll_path"])
            st.audio(true_piece_paths["mp3_path"])

        with cols[1]:
            # Quantized
            st.image(src_piece_paths["pianoroll_path"])
            st.audio(src_piece_paths["mp3_path"])

        with cols[2]:
            # Q.velocity ?
            st.image(qv_paths["pianoroll_path"])
            st.audio(qv_paths["mp3_path"])

        with cols[3]:
            # Predicted
            st.image(predicted_paths["pianoroll_path"])
            st.audio(predicted_paths["mp3_path"])


def tokenization_review_dashboard():
    st.markdown("### Quantization settings")

    n_dstart_bins = st.number_input(label="n_dstart_bins", value=3)
    n_duration_bins = st.number_input(label="n_duration_bins", value=3)
    n_velocity_bins = st.number_input(label="n_velocity_bins", value=3)

    quantizer = MidiQuantizer(
        n_dstart_bins=n_dstart_bins,
        n_duration_bins=n_duration_bins,
        n_velocity_bins=n_velocity_bins,
    )

    split = "train"
    dataset_name = "roszcz/maestro-v1-sustain"
    dataset = load_dataset(dataset_name, split=split)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Quantized")

    np.random.seed(137)
    n_samples = 5
    idxs = np.random.randint(len(dataset), size=n_samples)
    for idx in idxs:
        piece = MidiPiece.from_huggingface(dataset[int(idx)])
        quantized_piece = quantizer.quantize_piece(piece)

        av_dir = "tmp/dashboard/common"
        bins = f"{n_dstart_bins}-{n_duration_bins}-{n_velocity_bins}"
        save_name = f"{dataset_name}-{split}-{idx}".replace("/", "_")

        save_base_gt = os.path.join(av_dir, f"{save_name}")
        gt_paths = piece_av_files(piece, save_base=save_base_gt)

        save_base_quantized = os.path.join(av_dir, f"{save_name}-{bins}")
        quantized_piece_paths = piece_av_files(quantized_piece, save_base=save_base_quantized)

        st.json(piece.source)
        cols = st.columns(2)
        with cols[0]:
            st.image(gt_paths["pianoroll_path"])
            st.audio(gt_paths["mp3_path"])

        with cols[1]:
            st.image(quantized_piece_paths["pianoroll_path"])
            st.audio(quantized_piece_paths["mp3_path"])


def prepare_midi_pieces(
    record: dict,
    processed_df: pd.DataFrame,
    idx: int,
    dataset,
) -> tuple[MidiPiece, MidiPiece]:
    # get dataframes with notes
    quantized_notes = dataset.quantizer.apply_quantization(processed_df)

    # Same for the "src" piece
    start_time = np.min(processed_df["start"])
    processed_df["start"] -= start_time
    processed_df["end"] -= start_time

    quantized_piece = MidiPiece(quantized_notes)

    return quantized_piece


if __name__ == "__main__":
    main()
