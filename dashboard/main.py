import os
import glob
import json

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import streamlit as st
from fortepyan import MidiPiece
from omegaconf import OmegaConf, DictConfig

from model import make_model
from data.quantizer import MidiQuantizer
from data.dataset import load_cache_dataset
from dashboard.prompts import creative_prompts
from dashboard.predict_piece import predict_piece_dashboard
from utils import vocab_sizes, piece_av_files, generate_sequence

# Set the layout of the Streamlit page
st.set_page_config(layout="wide", page_title="Velocity Transformer", page_icon=":musical_keyboard")

with st.sidebar:
    devices = ["cpu"] + [f"cuda:{it}" for it in range(torch.cuda.device_count())]
    DEVICE = st.selectbox(label="Processing device", options=devices)


def main():
    with st.sidebar:
        dashboards = [
            "Creative Prompts",
            "Piece predictions",
            "Sequence predictions",
        ]
        mode = st.selectbox(label="Display", options=dashboards)

    with st.sidebar:
        # Show available checkpoints
        options = glob.glob("models/*.pt")
        options.sort()
        checkpoint_path = st.selectbox(label="model", options=options)
        st.markdown("Selected checkpoint:")
        st.markdown(checkpoint_path)

    # Load:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # - original config
    train_cfg = OmegaConf.create(checkpoint["cfg"])
    train_cfg.device = DEVICE

    # - - for model
    st.markdown("Model config:")
    model_params = OmegaConf.to_container(train_cfg.model)
    st.json(model_params, expanded=False)

    # - - for dataset
    dataset_params = OmegaConf.to_container(train_cfg.dataset)
    st.markdown("Dataset config:")
    st.json(dataset_params, expanded=True)

    # - model
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
    model.eval().to(DEVICE)

    # - quantizer
    quantizer = MidiQuantizer(
        n_dstart_bins=train_cfg.dataset.quantization.dstart,
        n_duration_bins=train_cfg.dataset.quantization.duration,
        n_velocity_bins=train_cfg.dataset.quantization.velocity,
    )
    st.markdown(f"Velocity bins: {quantizer.velocity_bin_edges}")

    n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    st.markdown(f"Model parameters: {n_parameters:.3f}M")

    # Folder to render audio and video
    model_dir = f"tmp/dashboard/{train_cfg.run_name}"

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if mode == "Piece predictions":
        predict_piece_dashboard(model, quantizer, train_cfg, model_dir)
    if mode == "Sequence predictions":
        model_predictions_review(model, quantizer, train_cfg, model_dir)
    if mode == "Creative Prompts":
        creative_prompts(model, quantizer, train_cfg, model_dir)


def model_predictions_review(
    model: nn.Module,
    quantizer: MidiQuantizer,
    train_cfg: DictConfig,
    model_dir: str,
):
    # load checkpoint, force dashboard device
    dataset_cfg = train_cfg.dataset
    dataset_name = st.text_input(label="dataset", value=train_cfg.dataset_name)
    split = st.text_input(label="split", value="test")

    random_seed = st.selectbox(label="random seed", options=range(20))

    dataset = load_cache_dataset(
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        split=split,
    )

    n_samples = 5
    np.random.seed(random_seed)
    idxs = np.random.randint(len(dataset), size=n_samples)

    pad_idx = dataset.src_encoder.token_to_id["<blank>"]

    cols = st.columns(3)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Q. velocity")
    with cols[2]:
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
            device=DEVICE,
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

        # create files
        true_save_base = os.path.join(model_dir, f"true_{record_id}")
        true_piece_paths = piece_av_files(piece=true_piece, save_base=true_save_base)

        qv_save_base = os.path.join(model_dir, f"qv_{record_id}")
        qv_paths = piece_av_files(piece=quantized_vel_piece, save_base=qv_save_base)

        predicted_save_base = os.path.join(model_dir, f"predicted_{record_id}")
        predicted_paths = piece_av_files(piece=pred_piece, save_base=predicted_save_base)

        # create a dashboard
        st.json(record_source)
        cols = st.columns(3)
        with cols[0]:
            # Unchanged
            st.image(true_piece_paths["pianoroll_path"])
            st.audio(true_piece_paths["mp3_path"])

        with cols[1]:
            # Q.velocity ?
            st.image(qv_paths["pianoroll_path"])
            st.audio(qv_paths["mp3_path"])

        with cols[2]:
            # Predicted
            st.image(predicted_paths["pianoroll_path"])
            st.audio(predicted_paths["mp3_path"])


if __name__ == "__main__":
    main()
