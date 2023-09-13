import glob

from tqdm import tqdm

from data.dataset import BinsToVelocityDataset
import fortepyan as ff
import torch
from evals.evaluate import load_checkpoint
from datasets import Dataset
import hydra
from omegaconf import DictConfig, OmegaConf
from model import make_model
from utils import predict_sample


@hydra.main(version_base=None, config_path="../config", config_name="process_conf")
def main(cfg: DictConfig):
    dev = torch.device(cfg.device)
    checkpoint = load_checkpoint(run_name=cfg.run_name, device=cfg.device)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    n_dstart_bins, n_duration_bins, n_velocity_bins = train_cfg.dataset.bins.split(" ")
    input_size = checkpoint["input_size"]
    output_size = checkpoint["output_size"]

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

    for file in glob.glob("midi_files/*.mid*"):
        piece = ff.MidiPiece.from_file(file)
        row = {
            "notes": piece.df,
            "midi_filename": file,
        }
        hf_dataset = Dataset.from_list([row])

        dataset = BinsToVelocityDataset(
            dataset=hf_dataset,
            n_dstart_bins=eval(n_dstart_bins),
            n_duration_bins=eval(n_duration_bins),
            n_velocity_bins=eval(n_velocity_bins),
            sequence_len=train_cfg.dataset.sequence_size,
        )
        predicted_tokens = []
        idx = 0
        for record in tqdm(dataset):
            idx += 1
            if idx % 2 == 0:
                continue
            sequence = predict_sample(record=record, dataset=dataset, model=model, cfg=cfg, train_cfg=train_cfg)
            predicted_tokens += sequence
        pred_velocities = dataset.tokenizer_tgt.untokenize(predicted_tokens)

        pred_df = piece.df.copy()
        pred_df = pred_df.head(len(pred_velocities))
        pred_df["velocity"] = pred_velocities

        pred_piece = ff.MidiPiece(pred_df)
        pred_piece.source = piece.source.copy()
        pred_path = file.split('.')[0] + "-pred.mid"
        pred_piece.to_midi().write(pred_path)


if __name__ == "__main__":
    main()