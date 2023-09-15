import glob

import hydra
import torch
import pandas as pd
import streamlit as st
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from fortepyan import MidiPiece
from datasets import load_dataset
from omegaconf import OmegaConf, DictConfig

from model import make_model
from data.dataset import BinsToVelocityDataset
from modules.label_smoothing import LabelSmoothing
from utils import avg_distance, piece_av_files, decode_and_output


@torch.no_grad()
def predict_piece_dashboard(cfg: DictConfig):
    with st.sidebar:
        model_path = st.selectbox(label="model", options=glob.glob("models/*.pt"))

    checkpoint = torch.load(model_path, map_location=cfg.device)
    params = pd.DataFrame(checkpoint["cfg"]["model"], index=[0])
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    st.markdown("Model parameters:")
    st.table(params)

    dev = torch.device(cfg.device)
    n_dstart_bins, n_duration_bins, n_velocity_bins = train_cfg.dataset.bins.split(" ")
    hf_dataset = load_dataset(cfg.dataset.dataset_name, split=cfg.dataset_split)

    if cfg.dataset.dataset_name == "roszcz/maestro-v1":
        pieces_names = zip(hf_dataset["composer"], hf_dataset["title"])
        with st.sidebar:
            composer, title = st.selectbox(
                label="piece",
                options=[composer + "    " + title for composer, title in pieces_names],
            ).split("    ")

        one_record_dataset = hf_dataset.filter(lambda x: x["composer"] == composer and x["title"] == title)
        midi_filename = composer + " " + title + ".mid"
    else:
        with st.sidebar:
            midi_filename = st.selectbox(label="piece", options=[filename for filename in hf_dataset["midi_filename"]])
        one_record_dataset = hf_dataset.filter(lambda x: x["midi_filename"] == midi_filename)

    dataset = BinsToVelocityDataset(
        dataset=one_record_dataset,
        n_dstart_bins=int(n_dstart_bins),
        n_duration_bins=int(n_duration_bins),
        n_velocity_bins=int(n_velocity_bins),
        sequence_len=train_cfg.dataset.sequence_size,
    )

    input_size = len(dataset.src_vocab)
    output_size = len(dataset.tgt_vocab)

    model = make_model(
        input_size=input_size,
        output_size=output_size,
        n=train_cfg.model.n,
        d_model=train_cfg.model.d_model,
        d_ff=train_cfg.model.d_ff,
        dropout=train_cfg.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)

    pad_idx = dataset.tgt_vocab.index("<blank>")

    criterion = LabelSmoothing(
        size=output_size,
        padding_idx=pad_idx,
        smoothing=train_cfg.train.label_smoothing,
    )
    criterion.to(dev)

    total_loss = 0
    total_dist = 0

    piece = MidiPiece.from_huggingface(one_record_dataset[0])

    piece.source["midi_filename"] = midi_filename

    predicted_piece_df = piece.df.copy()
    predicted_tokens = []
    idx = 0
    for record in tqdm(dataset):
        idx += 1
        if idx % 2 == 0:
            continue

        pad_idx = dataset.tgt_vocab.index("<blank>")
        src_mask = (record[0] != pad_idx).unsqueeze(-2)

        decoded, out = decode_and_output(
            model=model,
            src=record[0],
            src_mask=src_mask[0],
            max_len=train_cfg.dataset.sequence_size,
            start_symbol=0,
            device=cfg.device,
        )

        out_tokens = [dataset.tgt_vocab[x] for x in decoded if x != pad_idx]
        predicted_tokens += out_tokens

        target = record[1][1:-1].to(dev)
        n_tokens = (target != pad_idx).data.sum()
        loss = criterion(out, target) / n_tokens
        total_loss += loss.item()
        total_dist += avg_distance(out, target).cpu()

    pred_velocities = dataset.tokenizer_tgt.untokenize(predicted_tokens)
    predicted_piece_df = predicted_piece_df.head(len(pred_velocities))

    predicted_piece_df["velocity"] = pred_velocities.fillna(0)
    predicted_piece = MidiPiece(predicted_piece_df)

    predicted_piece.source = piece.source.copy()
    midi_filename = midi_filename.split(".")[0].replace("/", "-")
    predicted_piece.source["midi_filename"] = f"tmp/dashboard/{train_cfg.run_name}/{midi_filename}-pred.mid"
    piece.source["midi_filename"] = f"tmp/dashboard/common/{midi_filename}.mid"

    pred_paths = piece_av_files(predicted_piece)
    paths = piece_av_files(piece)

    cols = st.columns(2)

    # multiply by two because we use only half of the dataset samples
    avg_loss = 2 * total_loss / len(dataset)
    avg_dist = 2 * total_dist / len(dataset)
    predicted_piece.source["average_loss"] = f"{avg_loss:6.2f}"
    predicted_piece.source["average_dist"] = f"{avg_dist:6.2f}"

    print(f"{avg_loss:6.2f}, {avg_dist:6.2f}")

    with cols[0]:
        st.markdown("### True")
        st.image(paths["pianoroll_path"])
        st.audio(paths["mp3_path"])
        st.table(piece.source)
    with cols[1]:
        st.markdown("### Predicted")
        st.image(pred_paths["pianoroll_path"])
        st.audio(pred_paths["mp3_path"])
        st.table(predicted_piece.source)


@hydra.main(version_base=None, config_path="config", config_name="dashboard_conf")
def main(cfg):
    GlobalHydra.instance().clear()
    predict_piece_dashboard(cfg)


if __name__ == "__main__":
    main()
