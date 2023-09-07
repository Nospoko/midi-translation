from data.dataset import BinsToVelocityDataset
from evals import load_checkpoint, load_cached_dataset, greedy_decode
from utils import piece_av_files
from datasets import load_dataset
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
from data.batch import Batch
from model import make_model
from modules.label_smoothing import LabelSmoothing
import torch
from torch.utils.data import DataLoader
import einops
from utils import euclidean_distance
import pandas as pd
from fortepyan import MidiPiece
import streamlit as st


@hydra.main(version_base=None, config_path="config", config_name="render_conf")
@torch.no_grad()
def main(cfg):
    checkpoint = load_checkpoint(cfg.run_name, device=cfg.device)
    train_cfg = OmegaConf.create(checkpoint['cfg'])

    n_dstart_bins, n_duration_bins, n_velocity_bins = train_cfg.dataset.bins.split(' ')
    hf_dataset = load_dataset('roszcz/maestro-v1', split='test')
    for composer, title in zip(hf_dataset['composer'], hf_dataset['title']):
        print(composer, title)
    one_record_dataset = hf_dataset.filter(lambda x: x['composer'] == cfg.composer and x['title'] == cfg.title)
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
    model.to(cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    pad_idx = dataset.tgt_vocab.index('<blank>')

    criterion = LabelSmoothing(
        size=output_size,
        padding_idx=pad_idx,
        smoothing=train_cfg.train.label_smoothing,
    )
    criterion.to(cfg.device)

    total_tokens = 0
    total_loss = 0
    tokens = 0
    total_dist = 0

    dev = torch.device(cfg.device)
    dataloader = DataLoader(dataset, batch_size=1)

    piece = MidiPiece.from_huggingface(one_record_dataset[0])

    predicted_piece_df = piece.df.copy()
    predicted = torch.Tensor([])
    idx = 0
    for b in tqdm(dataloader):
        idx += 1
        if idx % 2 == 0:
            continue
        batch = Batch(b[0], b[1], pad=pad_idx)
        batch.to(dev)

        encoded_decoded = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encoded_decoded)

        out_rearranged = einops.rearrange(out, "b n d -> (b n) d")
        target = einops.rearrange(batch.tgt_y, "b n -> (b n)")
        loss = criterion(out_rearranged, target) / batch.ntokens
        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        total_dist += euclidean_distance(out_rearranged, target)
        predicted = torch.concat([predicted, out_rearranged])

    print(len(predicted))
    predicted = [dataset.tgt_vocab[x] for x in predicted.argmax(dim=1)]
    pred_velocities = dataset.tokenizer_tgt.untokenize(predicted)
    predicted_piece_df = predicted_piece_df.head(len(pred_velocities))

    predicted_piece_df['velocity'] = pred_velocities.fillna(0)
    predicted_piece = MidiPiece(predicted_piece_df)
    predicted_piece.source = piece.source.copy()
    predicted_piece.source['midi_filename'] = f"documentation/files/{cfg.composer}-{cfg.title}-{cfg.run_name}-pred.midi"
    piece.source["midi_filename"] = f"documentation/files/{piece.source['midi_filename'].replace('/', '-')}"
    pred_paths = piece_av_files(predicted_piece)
    paths = piece_av_files(piece)

    cols = st.columns(2)
    avg_loss = total_loss / len(dataset)
    avg_dist = total_dist / len(dataset)
    predicted_piece.source["average_loss"] = avg_loss
    predicted_piece.source["average_dist"] = avg_dist
    print(f"{avg_loss:6.2f}, {avg_dist:6.2f}")

    with cols[0]:
        st.image(paths["pianoroll_path"])
        st.audio(paths["mp3_path"])
        st.table(piece.source)
    with cols[1]:
        st.image(pred_paths["pianoroll_path"])
        st.audio(pred_paths["mp3_path"])
        st.table(predicted_piece.source)


if __name__ == '__main__':
    main()

