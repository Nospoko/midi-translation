import glob

import hydra
import torch
import einops
import pandas as pd
import streamlit as st
from tqdm import tqdm
from fortepyan import MidiPiece
from datasets import load_dataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

from data.batch import Batch
from model import make_model
from evals import greedy_decode
from data.dataset import BinsToVelocityDataset
from modules.label_smoothing import LabelSmoothing
from utils import piece_av_files, euclidean_distance


@torch.no_grad()
def predict_piece_dashboard(cfg: DictConfig):
    model_path = st.selectbox(label="model", options=glob.glob("models/*.pt"))

    checkpoint = torch.load(model_path, map_location=cfg.device)
    params = pd.DataFrame(checkpoint["cfg"]["model"], index=[0])
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    st.table(params)

    dev = torch.device(cfg.device)
    n_dstart_bins, n_duration_bins, n_velocity_bins = train_cfg.dataset.bins.split(" ")
    hf_dataset = load_dataset(cfg.dataset.dataset_name, split=cfg.dataset_split)
    if cfg.dataset.dataset_name == "roszcz/maestro-v1":
        pieces_names = zip(hf_dataset["composer"], hf_dataset["title"])
        composer, title = st.selectbox(
            label="piece",
            options=[composer + "    " + title for composer, title in pieces_names],
        ).split("    ")

        one_record_dataset = hf_dataset.filter(lambda x: x["composer"] == composer and x["title"] == title)
    else:
        midi_filename = st.selectbox(label="piece", options=[filename for filename in hf_dataset["midi_filename"]])
        one_record_dataset = hf_dataset.filter(lambda x: x["midi_filename"] == midi_filename)

    dataset = BinsToVelocityDataset(
        dataset=one_record_dataset,
        n_dstart_bins=eval(n_dstart_bins),
        n_duration_bins=eval(n_duration_bins),
        n_velocity_bins=eval(n_velocity_bins),
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

    dev = torch.device(dev)
    dataloader = DataLoader(dataset, batch_size=1)

    piece = MidiPiece.from_huggingface(one_record_dataset[0])

    predicted_piece_df = piece.df.copy()
    predicted = torch.Tensor([]).to(dev)
    idx = 0
    for b in tqdm(dataloader):
        idx += 1
        if idx % 2 == 0:
            continue
        batch = Batch(b[0], b[1], pad=pad_idx)
        batch.to(dev)
        sequence = greedy_decode(
            model=model,
            src=batch.src[0],
            src_mask=batch.src_mask[0],
            max_len=train_cfg.dataset.sequence_size,
            start_symbol=0,
            device=cfg.device,
        )
        predicted = torch.concat([predicted, sequence[1:]]).type_as(sequence.data)

        encoded_decoded = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encoded_decoded)

        out_rearranged = einops.rearrange(out, "b n d -> (b n) d")
        target = einops.rearrange(batch.tgt_y, "b n -> (b n)")
        loss = criterion(out_rearranged, target) / batch.ntokens
        total_loss += loss.item()
        total_dist += euclidean_distance(out_rearranged, target).cpu()

    predicted = [dataset.tgt_vocab[x] for x in predicted]
    pred_velocities = dataset.tokenizer_tgt.untokenize(predicted)
    predicted_piece_df = predicted_piece_df.head(len(pred_velocities))

    predicted_piece_df["velocity"] = pred_velocities.fillna(0)
    predicted_piece = MidiPiece(predicted_piece_df)
    predicted_piece.source = piece.source.copy()
    midi_filename = piece.source["midi_filename"].split('.')[0].replace("/", "-")
    predicted_piece.source["midi_filename"] = f"tmp/dashboard/{train_cfg.run_name}/{midi_filename}-whole_piece.midi"
    piece.source["midi_filename"] = f"tmp/dashboard/common/{midi_filename}.mid"
    pred_paths = piece_av_files(predicted_piece)
    paths = piece_av_files(piece)

    cols = st.columns(2)
    avg_loss = 2 * total_loss / len(dataset)
    avg_dist = 2 * total_dist / len(dataset)
    predicted_piece.source["average_loss"] = avg_loss
    predicted_piece.source["average_dist"] = avg_dist
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
    predict_piece_dashboard(cfg)


if __name__ == "__main__":
    main()
