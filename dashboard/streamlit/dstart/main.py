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
from predict_piece import predict_piece_dashboard

from model import make_model
from data.quantizer import MidiQuantizer
from data.tokenizer import DstartEncoder, QuantizedMidiEncoder
from utils import vocab_sizes, piece_av_files, generate_sequence
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset

# Set the layout of the Streamlit page
st.set_page_config(layout="wide", page_title="Dstart Transformer", page_icon=":musical_keyboard")

with st.sidebar:
    devices = ["cpu"] + [f"cuda:{it}" for it in range(torch.cuda.device_count())]
    DEVICE = st.selectbox(label="Processing device", options=devices)


@torch.no_grad()
def main():
    with st.sidebar:
        dashboards = [
            "Sequence predictions",
            "Piece predictions",
        ]
        mode = st.selectbox(label="Display", options=dashboards)

    with st.sidebar:
        # Show available checkpoints
        options = glob.glob("checkpoints/dstart/*.pt")
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
    st.markdown(f"Dstart bins: {quantizer.dstart_bin_edges}")

    n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    st.markdown(f"Model parameters: {n_parameters:.3f}M")

    # Folder to render audio and video
    model_dir = f"tmp/dashboard/{train_cfg.run_name}"

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if mode == "Sequence predictions":
        model_predictions_review(model, quantizer, train_cfg, model_dir)
    if mode == "Piece predictions":
        predict_piece_dashboard(model, quantizer, train_cfg, model_dir)


def model_predictions_review(
    model: nn.Module,
    quantizer: MidiQuantizer,
    train_cfg: DictConfig,
    model_dir: str,
):
    # load checkpoint, force dashboard device
    with st.form("select_dataset"):
        dataset_cfg = train_cfg.dataset
        dataset_name = st.text_input(label="dataset", value=train_cfg.dataset_name)
        split = st.text_input(label="split", value="test")
        random_seed = st.selectbox(label="random seed", options=range(20))
        run_button = st.form_submit_button(label="Run")
    if not run_button:
        return

    # load translation dataset and create MyTokenizedMidiDataset
    src_encoder = QuantizedMidiEncoder(quantization_cfg=train_cfg.dataset.quantization)
    tgt_encoder = DstartEncoder(n_bins=train_cfg.dstart_bins)
    translation_dataset = load_cache_dataset(
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        split=split,
    )
    dataset = MyTokenizedMidiDataset(
        dataset=translation_dataset,
        dataset_cfg=dataset_cfg,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )

    n_samples = 5
    np.random.seed(random_seed)
    idxs = np.random.randint(len(dataset), size=n_samples)

    cols = st.columns(3)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Q. Dstart")
    with cols[2]:
        st.markdown("### Predicted")

    # predict velocities and get src, tgt and model output
    print("Making predictions ...")
    for record_id in idxs:
        # Numpy to int :(
        record = dataset.get_complete_record(int(record_id))
        record_source = json.loads(record["source"])
        src_token_ids = record["source_token_ids"]

        generated_dstart = generate_sequence(
            model=model,
            device=DEVICE,
            src_tokens=src_token_ids,
            tgt_encoder=dataset.tgt_encoder,
            sequence_size=train_cfg.dataset.sequence_len,
        )

        # Just pitches and quantization n_bins of the source
        src_tokens = [dataset.src_encoder.vocab[token_id] for token_id in src_token_ids]
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
        quantized_dstart_df = true_piece.df.copy()

        # change untokenized velocities to model predictions
        pred_piece_df["start"] = tgt_encoder.unquantized_start(generated_dstart)
        pred_piece_df["end"] = pred_piece_df["start"] + pred_piece_df["duration"]

        quantized_dstart_df["start"] = quantized_piece.df["start"].copy()
        quantized_dstart_df["end"] = quantized_piece.df["start"] + true_piece.df["duration"]

        # create quantized piece with predicted velocities
        pred_piece = MidiPiece(pred_piece_df)
        quantized_vel_piece = MidiPiece(quantized_dstart_df)

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
