import torch
import gradio as gr
import fortepyan as ff
from omegaconf import OmegaConf
from fortepyan.audio import render as render_audio

from app.tools import load_model, predict_velocity


def run_app(target: str, midi_file):
    checkpoint = torch.load("checkpoints/velocity/midi-transformer-2023-10-03-08-34.pt", map_location="cpu")
    model = load_model(checkpoint)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    piece = ff.MidiPiece.from_file(midi_file.name)
    if target == "velocity":
        pred_piece = predict_velocity(model, train_cfg, piece)
    else:
        # TODO: predict_dstart
        pred_piece = predict_velocity(model, train_cfg, piece)

    audio_path = render_audio.midi_to_mp3(pred_piece.to_midi(), "tmp/predicted-audio.mp3")
    audio = gr.make_waveform(audio_path)

    return audio


def main():
    demo = gr.Interface(
        run_app,
        inputs=[
            gr.components.Radio(["velocity"]),
            gr.File(),
        ],
        outputs=[gr.Video()],
    )

    demo.launch()


if __name__ == "__main__":
    main()
