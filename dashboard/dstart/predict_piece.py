import os

import torch
import torch.nn as nn
import streamlit as st
from tqdm import tqdm
from datasets import Dataset
from fortepyan import MidiPiece
from omegaconf import DictConfig

from data.quantizer import MidiQuantizer
from modules.label_smoothing import LabelSmoothing
from data.tokenizer import DstartEncoder, QuantizedMidiEncoder
from dashboard.components import piece_selector, download_button
from data.dataset import MyTokenizedMidiDataset, quantized_piece_to_records
from utils import vocab_sizes, piece_av_files, decode_and_output, calculate_average_distance


@torch.no_grad()
def predict_piece_dashboard(
    model: nn.Module,
    quantizer: MidiQuantizer,
    train_cfg: DictConfig,
    model_dir: str,
):
    # Prepare everything required to make inference
    src_encoder = QuantizedMidiEncoder(train_cfg.dataset.quantization)
    tgt_encoder = DstartEncoder(n_bins=train_cfg.dstart_bins)

    piece, piece_descriptor = piece_selector(dataset_name=train_cfg.dataset_name)

    # And run full pre-processing ...
    qpiece = quantizer.inject_quantization_features(piece)

    # hard coded (idk if it needs changing)
    sequences = quantized_piece_to_records(
        piece=qpiece,
        sequence_len=128,
        sequence_step=128,
    )
    one_record_dataset = Dataset.from_list(sequences)

    # ... to get it into a format the model understands
    dataset = MyTokenizedMidiDataset(
        dataset=one_record_dataset,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
        dataset_cfg=train_cfg.dataset,
    )

    _, tgt_vocab_size = vocab_sizes(train_cfg)
    criterion = LabelSmoothing(
        size=tgt_vocab_size,
        smoothing=train_cfg.train.label_smoothing,
    )
    criterion.to(train_cfg.device)

    total_loss = 0
    total_dist = 0
    predicted_tokens = []
    for record in tqdm(dataset):
        src_token_ids = record["source_token_ids"]
        tgt_token_ids = record["target_token_ids"]

        predicted_token_ids, probabilities = decode_and_output(
            model=model,
            src=src_token_ids,
            max_len=128,
            device=train_cfg.device,
        )

        out_tokens = [tgt_encoder.vocab[x] for x in predicted_token_ids]
        predicted_tokens += out_tokens

        target = tgt_token_ids[1:].to(train_cfg.device)
        # minimal change to count tokens without padding idx
        n_tokens = (target != -1).data.sum()
        loss = criterion(probabilities, target) / n_tokens
        total_loss += loss.item()
        total_dist += calculate_average_distance(probabilities, target).cpu()

    predictions = tgt_encoder.untokenize(predicted_tokens)

    predicted_piece_df = piece.df.copy()
    predicted_piece_df = predicted_piece_df.head(len(predictions))

    predicted_piece_df["start"] = tgt_encoder.unquantized_start(predictions)
    predicted_piece_df["end"] = predicted_piece_df["start"] + predicted_piece_df["duration"]
    predicted_piece = MidiPiece(predicted_piece_df)

    predicted_piece.source = piece.source.copy()

    avg_loss = total_loss / len(dataset)
    avg_dist = total_dist / len(dataset)
    st.markdown(f"Average loss: {avg_loss}")
    st.markdown(f"Average distance: {avg_dist}")

    print(f"{avg_loss:6.2f}, {avg_dist:6.2f}")

    save_base_pred = f"{piece_descriptor}-{train_cfg.run_name}".replace("/", "_")
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
