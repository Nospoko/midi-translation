import os
import glob

import torch
import pandas as pd
import streamlit as st
from tqdm import tqdm
from fortepyan import MidiPiece
from omegaconf import OmegaConf
from datasets import Dataset, load_dataset

from model import make_model
from data.quantizer import MidiQuantizer
from dashboard.utils import download_button
from modules.label_smoothing import LabelSmoothing
from data.tokenizer import VelocityEncoder, QuantizedMidiEncoder
from data.dataset import MyTokenizedMidiDataset, quantized_piece_to_records
from utils import vocab_sizes, piece_av_files, decode_and_output, calculate_average_distance


@torch.no_grad()
def predict_piece_dashboard():
    dev = torch.device("cuda")

    with st.sidebar:
        model_path = st.selectbox(label="model", options=glob.glob("models/*.pt"))

    checkpoint = torch.load(model_path, map_location=dev)
    params = pd.DataFrame(checkpoint["cfg"]["model"], index=[0])
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    # Prepare everythin required to make inference
    quantizer = MidiQuantizer(
        n_dstart_bins=train_cfg.dataset.quantization.dstart,
        n_duration_bins=train_cfg.dataset.quantization.duration,
        n_velocity_bins=train_cfg.dataset.quantization.velocity,
    )
    src_encoder = QuantizedMidiEncoder(train_cfg.dataset.quantization)
    tgt_encoder = VelocityEncoder()

    st.markdown("Model parameters:")
    st.table(params)

    dataset_params = OmegaConf.to_container(train_cfg.dataset)
    st.markdown("Dataset config:")
    st.json(dataset_params, expanded=True)

    dataset_name = st.text_input(label="dataset", value=train_cfg.dataset_name)
    split = st.text_input(label="split", value="test")
    record_id = st.number_input(label="record id", value=0)
    hf_dataset = load_dataset(dataset_name, split=split)

    # Select one full piece
    record = hf_dataset[record_id]
    piece = MidiPiece.from_huggingface(record)

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

    src_vocab_size, tgt_vocab_size = vocab_sizes(train_cfg)
    model = make_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        n=train_cfg.model.n,
        d_model=train_cfg.model.d_model,
        d_ff=train_cfg.model.d_ff,
        dropout=train_cfg.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)

    pad_idx = src_encoder.token_to_id["<blank>"]

    criterion = LabelSmoothing(
        size=tgt_vocab_size,
        padding_idx=pad_idx,
        smoothing=train_cfg.train.label_smoothing,
    )
    criterion.to(dev)

    total_loss = 0
    total_dist = 0

    predicted_tokens = []
    for record in tqdm(dataset):
        src_token_ids = record["source_token_ids"]
        tgt_token_ids = record["target_token_ids"]
        src_mask = (src_token_ids != pad_idx).unsqueeze(-2)

        predicted_token_ids, probabilities = decode_and_output(
            model=model,
            src=src_token_ids,
            src_mask=src_mask[0],
            max_len=train_cfg.dataset.sequence_len,
            start_symbol=0,
            device=dev,
        )

        out_tokens = [tgt_encoder.vocab[x] for x in predicted_token_ids if x != pad_idx]
        predicted_tokens += out_tokens

        target = tgt_token_ids[1:-1].to(dev)
        n_tokens = (target != pad_idx).data.sum()
        loss = criterion(probabilities, target) / n_tokens
        total_loss += loss.item()
        total_dist += calculate_average_distance(probabilities, target).cpu()

    pred_velocities = tgt_encoder.untokenize(predicted_tokens)

    predicted_piece_df = piece.df.copy()
    predicted_piece_df = predicted_piece_df.head(len(pred_velocities))

    predicted_piece_df["velocity"] = pred_velocities.fillna(0)
    predicted_piece = MidiPiece(predicted_piece_df)

    predicted_piece.source = piece.source.copy()

    # multiply by two because we use only half of the dataset samples
    avg_loss = 2 * total_loss / len(dataset)
    avg_dist = 2 * total_dist / len(dataset)
    predicted_piece.source["average_loss"] = f"{avg_loss:6.2f}"
    predicted_piece.source["average_dist"] = f"{avg_dist:6.2f}"

    print(f"{avg_loss:6.2f}, {avg_dist:6.2f}")

    # Render audio and video
    model_dir = f"tmp/dashboard/{train_cfg.run_name}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    save_base_pred = f"{dataset_name}-{split}-{record_id}".replace("/", "_")
    save_base_pred = os.path.join(model_dir, save_base_pred)
    pred_paths = piece_av_files(predicted_piece, save_base=save_base_pred)

    save_base_gt = save_base_pred + "-gt"
    gt_paths = piece_av_files(piece, save_base=save_base_gt)

    st.json(piece.source)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Original")
        st.image(gt_paths["pianoroll_path"])
        st.audio(gt_paths["mp3_path"])

        midi_path = gt_paths["midi_path"]
        with open(midi_path, "rb") as file:
            download_button_str = download_button(
                object_to_download=file.read(),
                download_filename=midi_path.split("/")[-1],
                button_text="Download original midi",
            )
            st.markdown(download_button_str, unsafe_allow_html=True)

    with cols[1]:
        st.markdown("### Generated")
        st.image(pred_paths["pianoroll_path"])
        st.audio(pred_paths["mp3_path"])

        midi_path = pred_paths["midi_path"]
        with open(midi_path, "rb") as file:
            download_button_str = download_button(
                object_to_download=file.read(),
                download_filename=midi_path.split("/")[-1],
                button_text="Download generated midi",
            )
            st.markdown(download_button_str, unsafe_allow_html=True)


# TODO Move this to some kind of dashboard utils
