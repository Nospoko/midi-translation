import os

import torch
import numpy as np
import pandas as pd
import streamlit as st
from fortepyan import MidiPiece
from omegaconf import OmegaConf
from datasets import load_dataset

from model import make_model
from utils import piece_av_files
from data.dataset import BinsToVelocityDataset
from eval import make_examples, load_test_dataset


def main():
    mode = st.selectbox(label="Display", options=["Model predictions", "Tokenization review"])
    if mode == "Tokenization review":
        tokenization_review_dashboard()
    if mode == "Model predictions":
        model_predictions_review()


def get_sample_info(dataset: BinsToVelocityDataset, midi_filename: str):
    sample_data = dataset.dataset.filter(lambda row: row["midi_filename"] == midi_filename)
    title, composer = sample_data["title"][0], sample_data["composer"][0]
    return title, composer


def model_predictions_review():
    # options
    path = "models/" + st.selectbox(label="model", options=os.listdir("models"))
    start_index = eval(st.text_input(label="start index", value="0"))

    # load checkpoint
    checkpoint = torch.load(path, map_location="cpu")
    cols = st.columns(4)

    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Quantized")
    with cols[2]:
        st.markdown("### Target")
    with cols[3]:
        st.markdown("### Predicted")

    train_cfg = OmegaConf.create(checkpoint["cfg"])

    dataset = load_dataset(train_cfg.dataset)

    model = make_model(
        input_size=len(dataset.src_vocab),
        output_size=len(dataset.tgt_vocab),
        n=train_cfg.model.n,
        d_ff=train_cfg.model.d_ff,
        h=train_cfg.model.h,
        dropout=train_cfg.model.dropout,
    )

    n_samples = 5
    # predict velocities and get src, tgt and model output
    results = make_examples(dataset=dataset, model=model, start_index=start_index, n_examples=n_samples)

    bins = train_cfg.bins.replace(" ", "-")
    for it in range(n_samples):
        # I use every second record, so as not to create overlapped examples - it works together with make_examples()
        idx = it * 2

        src = results[it]["src"]
        tgt = results[it]["tgt"]
        out = results[it]["out"]

        # get unprocessed data
        record = dataset.unprocessed_records[idx + start_index]

        # get untokenized source data
        source = dataset.tokenizer_src.untokenize(src)
        predicted = dataset.tokenizer_tgt.untokenize(out)
        velocities = dataset.tokenizer_tgt.untokenize(tgt)
        print("start")

        filename = record["midi_filename"]

        # prepare unprocessed and tokenized midi pieces
        true_piece, src_piece = prepare_midi_pieces(record, source, idx=idx + start_index, dataset=dataset, bins=bins)
        pred_piece_df = src_piece.df.copy()
        tgt_piece_df = src_piece.df.copy()

        # change untokenized velocities to model predictions
        # TODO: predictions are sometimes the length of 127 or 126 instead of 128 ???
        tgt_piece_df["velocity"] = velocities
        pred_piece_df["velocity"] = predicted
        pred_piece_df["velocity"] = pred_piece_df["velocity"].fillna(0)

        # create quantized piece with predicted velocities
        pred_piece = MidiPiece(pred_piece_df)
        tgt_piece = MidiPiece(tgt_piece_df)

        pred_piece.source = true_piece.source.copy()
        tgt_piece.source = true_piece.source.copy()

        name = filename.split("/")[0] + "/" + str(idx + start_index) + "-predicted-" + bins
        pred_piece.source["midi_filename"] = name + os.path.basename(filename)

        name = filename.split("/")[0] + "/" + str(idx + start_index) + "-target-" + bins
        tgt_piece.source["midi_filename"] = name + os.path.basename(filename)

        print("Creating files ...")
        # create files
        paths = piece_av_files(true_piece)
        src_piece_paths = piece_av_files(src_piece)
        predicted_paths = piece_av_files(pred_piece)
        tgt_paths = piece_av_files(tgt_piece)

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
            st.image(tgt_paths["pianoroll_path"])
            st.audio(tgt_paths["mp3_path"])
            st.table(tgt_piece.source)

        with cols[3]:
            st.image(predicted_paths["pianoroll_path"])
            st.audio(predicted_paths["mp3_path"])
            st.table(pred_piece.source)


def tokenization_review_dashboard():
    st.markdown("### Tokenization method:\n" "**n_dstart_bins    n_duration_bins    n_velocity_bins**")
    bins = st.text_input(label="bins", value="3 3 3")
    dataset_cfg = OmegaConf.create({"bins": bins, "sequence_size": 128})

    dataset = load_test_dataset(dataset_cfg)
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
            unprocessed=dataset.unprocessed_records[idx],
            processed=dataset.processed_records[idx],
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


def prepare_midi_pieces(
    unprocessed: dict, processed: dict, idx: int, dataset: BinsToVelocityDataset, bins: str = "3-3-3"
) -> tuple[MidiPiece, MidiPiece]:
    # get dataframes with notes
    processed_df = pd.DataFrame(processed)

    notes = pd.DataFrame(unprocessed)
    quantized_notes = dataset.quantizer.apply_quantization(processed_df)
    # we have to pop midi_filename column
    filename = notes.pop("midi_filename")[0]
    # print(filename)
    title, composer = get_sample_info(dataset=dataset, midi_filename=filename)

    start_time = np.min(notes["start"])

    # normalize start and end time
    notes["start"] -= start_time
    notes["end"] -= start_time
    start_time = np.min(processed_df["start"])
    processed_df["start"] -= start_time
    processed_df["end"] -= start_time

    # create MidiPieces
    piece = MidiPiece(notes)
    name = filename.split("/")[0] + "/" + str(idx) + "-real-" + bins
    piece.source["midi_filename"] = name + os.path.basename(filename)
    piece.source["title"] = title
    piece.source["composer"] = composer

    quantized_piece = MidiPiece(quantized_notes)
    name = filename.split("/")[0] + "/" + str(idx) + "-quantized-" + bins
    quantized_piece.source["midi_filename"] = name + os.path.basename(filename)
    quantized_piece.source["title"] = title
    quantized_piece.source["composer"] = composer

    return piece, quantized_piece


if __name__ == "__main__":
    main()
