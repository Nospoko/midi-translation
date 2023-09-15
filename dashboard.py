import os
import glob
import json

import hydra
import torch
import numpy as np
import pandas as pd
import streamlit as st
from fortepyan import MidiPiece
from omegaconf import OmegaConf, DictConfig
from hydra.core.global_hydra import GlobalHydra

from model import make_model
from evals.evaluate import load_cached_dataset
from utils import piece_av_files, predict_sample
from predict_piece import predict_piece_dashboard
from data.dataset import BinsToDstartDataset, TokenizedMidiDataset, BinsToVelocityDataset

@hydra.main(version_base=None, config_path="config", config_name="dashboard_conf")
def main(cfg: DictConfig):
    with st.sidebar:
        mode = st.selectbox(label="Display", options=["Model predictions", "Predict piece", "Tokenization review"])
    if mode == "Tokenization review":
        tokenization_review_dashboard(cfg)
    if mode == "Predict piece":
        predict_piece_dashboard(cfg)
    if mode == "Model predictions":
        model_predictions_review(cfg)


def get_sample_info(dataset: BinsToVelocityDataset, midi_filename: str):
    sample_data = dataset.dataset.filter(lambda row: row["midi_filename"] == midi_filename)
    title, composer = sample_data["title"][0], sample_data["composer"][0]
    return title, composer


def model_predictions_review(cfg: DictConfig):
    with st.sidebar:
        # options
        path = st.selectbox(label="model", options=glob.glob("models/*.pt"))

    # load checkpoint
    checkpoint = torch.load(path, map_location=cfg.device)
    params = pd.DataFrame(checkpoint["cfg"]["model"], index=[0])
    train_cfg = OmegaConf.create(checkpoint["cfg"])
    st.markdown("Model parameters:")
    st.table(params)

    cols = st.columns(4)

    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Quantized")
    with cols[2]:
        st.markdown("### Q. column")
    with cols[3]:
        st.markdown("### Predicted")

    model, dataset = prepare_model_and_dataset_from_checkpoint(checkpoint, cfg)

    n_samples = 5
    idxs = np.random.randint(len(dataset), size=n_samples)

    records = [dataset.records[idx] for idx in idxs]
    samples = [dataset[idx] for idx in idxs]

    pad_idx = dataset.tgt_vocab.index("<blank>")
    bins = train_cfg.dataset.bins.replace(" ", "-")

    # predict velocities and get src, tgt and model output
    print("Making predictions ...")
    for record, sample, idx in zip(records, samples, idxs):
        result = predict_sample(
            record=sample,
            dataset=dataset,
            model=model,
            cfg=cfg,
            train_cfg=train_cfg,
        )
        src = [dataset.src_vocab[x] for x in sample[0] if x != pad_idx]

        out = result
        record["source"] = json.loads(record["source"])

        source = dataset.tokenizer_src.untokenize(src)
        predicted = dataset.tokenizer_tgt.untokenize(out)
        filename = record["midi_filename"]

        true_piece, src_piece = prepare_midi_pieces(record, source, idx=idx, dataset=dataset, bins=bins)

        if type(dataset) == BinsToVelocityDataset:
            predicted_column = "velocity"
            pred_piece_df = true_piece.df.copy()
            # change untokenized velocities to model predictions
            pred_piece_df["velocity"] = predicted
            pred_piece_df["velocity"] = pred_piece_df["velocity"].fillna(0)
        elif type(dataset) == BinsToDstartDataset:
            predicted_column = "dstart_bin"

            pred_piece_df = source.copy()
            pred_piece_df["dstart_bin"] = predicted.astype("int16")
            # get quantized df with predictions
            dataset.quantizer.apply_quantization_with_tgt_bins(pred_piece_df)
            # make df like an original but with predicted dstart
            pred_piece_df[["velocity", "duration"]] = true_piece.df[["velocity", "duration"]].copy()
            pred_piece_df["end"] = pred_piece_df["start"] + pred_piece_df["duration"]

        pred_piece = MidiPiece(pred_piece_df)

        # create quantized piece with predicted velocities
        quantized_col_df = true_piece.df.copy()
        quantized_col_df[predicted_column] = src_piece.df[predicted_column].copy()

        quantized_vel_piece = MidiPiece(quantized_col_df)

        pred_piece.source = true_piece.source.copy()
        quantized_vel_piece.source = true_piece.source.copy()

        model_dir = f"tmp/dashboard/{train_cfg.run_name}"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        directory = "tmp/dashboard/"

        name = f"{filename.split('.')[0].replace('/', '-')}-{idx}"
        pred_piece.source = true_piece.source.copy()
        pred_piece.source["midi_filename"] = model_dir + "/" + name + ".mid"

        name = f"{filename.split('.')[0].replace('/', '-')}-{idx}-qv-{bins}-{dataset.sequence_len}"
        quantized_vel_piece.source = true_piece.source.copy()
        quantized_vel_piece.source["midi_filename"] = directory + "common/" + name + ".mid"

        print("Creating files ...")
        # create files
        paths = piece_av_files(true_piece)
        src_piece_paths = piece_av_files(src_piece)
        qv_paths = piece_av_files(quantized_vel_piece)
        predicted_paths = piece_av_files(pred_piece)

        # create a dashboard
        with cols[0]:
            st.image(paths["pianoroll_path"])
            st.audio(paths["mp3_path"])
            st.table(true_piece.source)

        with cols[1]:
            st.image(src_piece_paths["pianoroll_path"])
            st.audio(src_piece_paths["mp3_path"])
            st.table(src_piece.source)

        with cols[2]:
            st.image(qv_paths["pianoroll_path"])
            st.audio(qv_paths["mp3_path"])
            st.table(quantized_vel_piece.source)

        with cols[3]:
            st.image(predicted_paths["pianoroll_path"])
            st.audio(predicted_paths["mp3_path"])
            st.table(pred_piece.source)


def tokenization_review_dashboard(cfg):
    st.markdown("### Tokenization method:\n" "**n_dstart_bins    n_duration_bins    n_velocity_bins**")
    bins = st.text_input(label="bins", value="3 3 3")
    dataset_cfg = OmegaConf.create(
        {
            "dataset_class": cfg.dataset.dataset_class,
            "dataset_name": "roszcz/maestro-v1",
            "bins": bins,
            "sequence_size": 128,
            "n_tgt_dstart_bins": None,
        }
    )

    dataset = load_cached_dataset(dataset_cfg)
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
            bins=bins,
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


def prepare_model_and_dataset_from_checkpoint(
    checkpoint: dict,
    cfg: DictConfig,
) -> tuple[torch.nn.Module, TokenizedMidiDataset]:
    train_cfg = OmegaConf.create(checkpoint["cfg"])
    dataset_name = cfg.dataset.dataset_name
    if dataset_name is None:
        dataset = load_cached_dataset(train_cfg.dataset)
    else:
        cfg.dataset = train_cfg.dataset
        cfg.dataset.dataset_name = dataset_name
        dataset = load_cached_dataset(cfg.dataset, split=cfg.dataset_split)

    model = make_model(
        input_size=len(dataset.src_vocab),
        output_size=len(dataset.tgt_vocab),
        n=train_cfg.model.n,
        d_model=train_cfg.model.d_model,
        d_ff=train_cfg.model.d_ff,
        h=train_cfg.model.h,
        dropout=train_cfg.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.device)

    return model, dataset


def prepare_midi_pieces(
    record: dict, processed: dict, idx: int, dataset: TokenizedMidiDataset, bins: str = "3-3-3"
) -> tuple[MidiPiece, MidiPiece]:
    # get dataframes with notes
    processed_df = pd.DataFrame(processed)
    piece_source = record.pop("source")
    notes = pd.DataFrame(record)
    quantized_notes = dataset.quantizer.apply_quantization(processed_df)
    # we have to pop midi_filename column
    filename = notes.pop("midi_filename")[0]

    start_time = np.min(notes["start"])

    # normalize start and end time
    notes["start"] -= start_time
    notes["end"] -= start_time
    start_time = np.min(processed_df["start"])
    processed_df["start"] -= start_time
    processed_df["end"] -= start_time

    # create MidiPieces
    piece = MidiPiece(notes)
    piece.source = piece_source

    name = f"{filename.split('.')[0].replace('/', '-')}-{idx}-real-{bins}-{dataset.sequence_len}"
    piece.source["midi_filename"] = f"tmp/dashboard/common/{name}.mid"

    quantized_piece = MidiPiece(quantized_notes)

    name = f"{filename.split('.')[0].replace('/', '-')}-{idx}-quantized-{bins}-{dataset.sequence_len}"
    quantized_piece.source = piece.source.copy()
    quantized_piece.source["midi_filename"] = f"tmp/dashboard/common/{name}.mid"

    return piece, quantized_piece


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
