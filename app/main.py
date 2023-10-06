import glob

import torch
import gradio as gr
import fortepyan as ff
from omegaconf import OmegaConf
from fortepyan.audio import render as render_audio

from app.tools import load_model, predict_dstart, predict_velocity


def run_dstart_app(midi_file, model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = load_model(checkpoint)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    piece = ff.MidiPiece.from_file(midi_file.name)
    predicted_piece = predict_dstart(model, train_cfg, piece)

    audio_path = render_audio.midi_to_mp3(predicted_piece.to_midi(), "tmp/predicted-audio.mp3")
    audio = gr.make_waveform(audio_path)

    return audio


def run_velocity_app(midi_file, model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = load_model(checkpoint)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    piece = ff.MidiPiece.from_file(midi_file.name)
    predicted_piece = predict_velocity(model, train_cfg, piece)

    audio_path = render_audio.midi_to_mp3(predicted_piece.to_midi(), "tmp/predicted-audio.mp3")
    audio = gr.make_waveform(audio_path)

    return audio


def main():
    with gr.Blocks() as demo:
        file = (gr.File(file_count="single", label="midi file"),)
        with gr.Tab("Predict Velocity"):
            with gr.Row():
                with gr.Column():
                    velocity_model_path = gr.Dropdown(glob.glob("checkpoints/velocity/*.pt"), label="model")
                    velocity_button = gr.Button("Predict")
                with gr.Column():
                    velocity_out = gr.Video()

        with gr.Tab("Predict Dstart"):
            with gr.Row():
                with gr.Column():
                    dstart_model_path = gr.Dropdown(glob.glob("checkpoints/dstart/*.pt"), label="model")
                    dstart_button = gr.Button("Predict")
                with gr.Column():
                    dstart_out = gr.Video()

        velocity_button.click(run_velocity_app, inputs=[file[0], velocity_model_path], outputs=velocity_out)
        dstart_button.click(run_dstart_app, inputs=[file[0], dstart_model_path], outputs=dstart_out)

    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
