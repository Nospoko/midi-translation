import glob

import torch
import numpy as np
import gradio as gr
import fortepyan as ff
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
from fortepyan.audio import render as render_audio

from app.tools import load_model, predict_dstart, predict_velocity


def run_dstart_app(midi_file, model_path: str, progress=gr.Progress(track_tqdm=True)):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = load_model(checkpoint)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    piece = ff.MidiPiece.from_file(midi_file.name)
    predicted_piece = predict_dstart(model, train_cfg, piece)

    audio_path = render_audio.midi_to_mp3(predicted_piece.to_midi(), "tmp/predicted-audio.mp3")
    audio = gr.make_waveform(audio_path)

    fig = ff.view.draw_pianoroll_with_velocities(predicted_piece)
    plt.tight_layout()

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return audio, image_from_plot


def run_velocity_app(midi_file, model_path: str, progress=gr.Progress(track_tqdm=True)):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = load_model(checkpoint)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    piece = ff.MidiPiece.from_file(midi_file.name)
    predicted_piece = predict_velocity(model, train_cfg, piece)

    audio_path = render_audio.midi_to_mp3(predicted_piece.to_midi(), "tmp/predicted-audio.mp3")
    audio = gr.make_waveform(audio_path)

    fig = ff.view.draw_pianoroll_with_velocities(predicted_piece)
    plt.tight_layout()

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return audio, image_from_plot


def main():
    with gr.Blocks() as demo:
        file = (gr.File(file_count="single", label="midi file"),)
        with gr.Tab("Predict Velocity"):
            with gr.Row():
                with gr.Column():
                    velocity_model_path = gr.Dropdown(glob.glob("checkpoints/velocity/*.pt"), label="model")
                    velocity_button = gr.Button("Predict")
                with gr.Column():
                    velocity_pianoroll = gr.Image(label="piano_roll")
                    velocity_out = gr.Video(label="predicted_piece")

        with gr.Tab("Predict Dstart"):
            with gr.Row():
                with gr.Column():
                    dstart_model_path = gr.Dropdown(glob.glob("checkpoints/dstart/*.pt"), label="model")
                    dstart_button = gr.Button("Predict")
                with gr.Column():
                    dstart_pianoroll = gr.Image(label="piano_roll")
                    dstart_out = gr.Video(label="predicted_piece")

        velocity_button.click(
            run_velocity_app,
            inputs=[file[0], velocity_model_path],
            outputs=[velocity_out, velocity_pianoroll],
        )
        dstart_button.click(
            run_dstart_app,
            inputs=[file[0], dstart_model_path],
            outputs=[dstart_out, dstart_pianoroll],
        )

    demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
