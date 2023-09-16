import os
import glob
import json

import torch
import numpy as np
import pandas as pd
import streamlit as st
from fortepyan import MidiPiece
from omegaconf import OmegaConf

from model import make_model
from data.dataset import BinsToVelocityDataset
from evals.evaluate import load_cached_dataset
from predict_piece import predict_piece_dashboard
from utils import piece_av_files, generate_sequence, vocab_sizes


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


def get_sample_info(dataset: BinsToVelocityDataset, midi_filename: str):
    sample_data = dataset.dataset.filter(lambda row: row["midi_filename"] == midi_filename)
    title, composer = sample_data["title"][0], sample_data["composer"][0]
    return title, composer


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
    dataset_name = st.text_input(label="dataset", value=dataset_cfg.dataset_name)
    split = st.text_input(label="split", value="test")

    random_seed = st.selectbox(label="random seed", options=range(20))

    dataset_cfg.dataset_name = dataset_name
    dataset = load_cached_dataset(dataset_cfg=dataset_cfg, split=split)
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

    pad_idx = dataset.tgt_vocab.index("<blank>")

    cols = st.columns(4)
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
        record = dataset.records[record_id]
        record_source = json.loads(record["source"])

        record_token_ids = dataset[record_id]

        src_token_ids = record_token_ids[0]
        generated_tokens = generate_sequence(
            model=model,
            dataset=dataset,
            src_tokens=src_token_ids,
            sequence_size=train_cfg.dataset.sequence_size,
        )
        generated_velocity = dataset.tokenizer_tgt.untokenize(generated_tokens)

        # Just pitches and quantization bins of the source
        src_tokens = [dataset.src_vocab[x] for x in src_token_ids if x != pad_idx]
        source_df = dataset.tokenizer_src.untokenize(src_tokens)
        quantized_notes = dataset.quantizer.apply_quantization(source_df)
        quantized_piece = MidiPiece(quantized_notes)
        quantized_piece.time_shift(-quantized_piece.df.start.min())

        # Reconstruct the sequence as recorded
        true_notes = pd.DataFrame(record)
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
    st.markdown("### Tokenization method:\n" "**n_dstart_bins    n_duration_bins    n_velocity_bins**")
    bins = st.text_input(label="bins", value="3 3 3")
    dataset_cfg = OmegaConf.create({"dataset_name": "roszcz/maestro-v1-sustain", "bins": bins, "sequence_size": 128})

    dataset = load_cached_dataset(dataset_cfg=dataset_cfg)
    bins = bins.replace(" ", "-")

    n_samples = 5
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Quantized")

    indexes = torch.randint(0, len(dataset), [n_samples])
    for idx in indexes:
        piece, quantized_piece = prepare_midi_pieces(
            record=dataset.records[idx],
            processed=dataset.records[idx],
            idx=idx,
            dataset=dataset,
        )

        paths = piece_av_files(piece)
        quantized_piece_paths = piece_av_files(quantized_piece)

        with cols[0]:
            st.image(paths["pianoroll_path"])
            st.audio(paths["mp3_path"])
            st.table(piece.source)

        with cols[1]:
            st.image(quantized_piece_paths["pianoroll_path"])
            st.audio(quantized_piece_paths["mp3_path"])
            st.table(quantized_piece.source)


def prepare_midi_pieces(
    record: dict,
    processed_df: pd.DataFrame,
    idx: int,
    dataset: BinsToVelocityDataset,
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
