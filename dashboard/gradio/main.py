import os
import glob

import torch
import gradio as gr
import fortepyan as ff
from omegaconf import OmegaConf

from dashboard.gradio.tools import load_model, predict_dstart, predict_velocity, make_pianoroll_video


def run_dstart_app(midi_file, model_path: str, progress=gr.Progress(track_tqdm=True)):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = load_model(checkpoint)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    piece = ff.MidiPiece.from_file(midi_file.name)
    predicted_piece = predict_dstart(model, train_cfg, piece)

    vid = make_pianoroll_video(predicted_piece)

    file_name = f"tmp/predicted-{os.path.basename(midi_file.name)}"
    predicted_piece.to_midi().write(file_name)

    return vid, file_name


def run_velocity_app(midi_file, model_path: str, progress=gr.Progress(track_tqdm=True)):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = load_model(checkpoint)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    piece = ff.MidiPiece.from_file(midi_file.name)
    predicted_piece = predict_velocity(model, train_cfg, piece)

    vid = make_pianoroll_video(predicted_piece)

    file_name = f"tmp/predicted-{os.path.basename(midi_file.name)}"
    predicted_piece.to_midi().write(file_name)

    return vid, file_name


def main():
    with gr.Blocks() as demo:
        file = (gr.File(file_count="single", label="midi_file"),)
        with gr.Tab("Predict Velocity"):
            with gr.Row():
                with gr.Column():
                    velocity_model_path = gr.Dropdown(glob.glob("checkpoints/velocity/*.pt"), label="model")
                    velocity_button = gr.Button("Predict")
                with gr.Column():
                    velocity_out = gr.Video(label="predicted_piece")
                    velocity_file = gr.File(label="predicted_midi")

        with gr.Tab("Predict Dstart"):
            with gr.Row():
                with gr.Column():
                    dstart_model_path = gr.Dropdown(glob.glob("checkpoints/dstart/*.pt"), label="model")
                    dstart_button = gr.Button("Predict")
                with gr.Column():
                    dstart_out = gr.Video(label="predicted_piece")
                    dstart_file = gr.File(label="predicted_midi")

        velocity_button.click(
            run_velocity_app,
            inputs=[file[0], velocity_model_path],
            outputs=[velocity_out, velocity_file],
        )
        dstart_button.click(
            run_dstart_app,
            inputs=[file[0], dstart_model_path],
            outputs=[dstart_out, dstart_file],
        )

    demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
