import os
import glob

import torch
import gradio as gr
import fortepyan as ff
from omegaconf import OmegaConf

from dashboard.gradio.tools import load_model, predict_dstart, quantize_piece, predict_velocity, make_pianoroll_video


def run_predict_app(midi_file, dstart_model_path: str, velocity_model_path: str, progress=gr.Progress(track_tqdm=True)):
    dstart_checkpoint = torch.load(dstart_model_path, map_location="cpu")
    dstart_model = load_model(dstart_checkpoint)
    dstart_cfg = OmegaConf.create(dstart_checkpoint["cfg"])

    velocity_checkpoint = torch.load(velocity_model_path, map_location="cpu")
    velocity_model = load_model(velocity_checkpoint)
    velocity_cfg = OmegaConf.create(velocity_checkpoint["cfg"])

    piece = ff.MidiPiece.from_file(midi_file.name)
    original_vid = make_pianoroll_video(piece)

    # seperate function to create quantized piece for better clarity
    # might think of a way to quantize only one time e.g. pass qpiece instead of piece to predict_velocity/dstart
    qpiece = quantize_piece(dstart_cfg, piece)
    quantized_vid = make_pianoroll_video(qpiece)

    pred_dstart = predict_dstart(dstart_model, dstart_cfg, piece)
    pred_velocity = predict_velocity(velocity_model, velocity_cfg, piece)

    predicted_piece_df = pred_dstart.df.copy()
    predicted_piece_df["velocity"] = pred_velocity.df["velocity"]

    predicted_piece = ff.MidiPiece(predicted_piece_df)

    vid = make_pianoroll_video(predicted_piece)

    file_name = f"tmp/pred-{os.path.basename(midi_file.name)}"
    predicted_piece.to_midi().write(file_name)

    return original_vid, quantized_vid, vid, file_name


def run_dstart_app(midi_file, model_path: str, progress=gr.Progress(track_tqdm=True)):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = load_model(checkpoint)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    # Folder to render midi files
    model_dir = f"tmp/dashboard/{train_cfg.run_name}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    piece = ff.MidiPiece.from_file(midi_file.name)
    original_vid = make_pianoroll_video(piece)

    # seperate function to create quantized piece for better clarity
    # might think of a way to quantize only one time e.g. pass qpiece instead of piece to predict_velocity/dstart
    qpiece = quantize_piece(train_cfg, piece)
    quantized_vid = make_pianoroll_video(qpiece)

    predicted_piece = predict_dstart(model, train_cfg, piece)

    vid = make_pianoroll_video(predicted_piece)

    file_name = f"{model_dir}/pred-{os.path.basename(midi_file.name)}"
    predicted_piece.to_midi().write(file_name)

    return original_vid, quantized_vid, vid, file_name


def run_velocity_app(midi_file, model_path: str, progress=gr.Progress(track_tqdm=True)):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = load_model(checkpoint)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    # Folder to render midi files
    model_dir = f"tmp/dashboard/{train_cfg.run_name}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    piece = ff.MidiPiece.from_file(midi_file.name)
    original_vid = make_pianoroll_video(piece)

    # seperate function to create quantized piece for better clarity
    # might think of a way to quantize only one time e.g. pass qpiece instead of piece to predict_velocity/dstart
    qpiece = quantize_piece(train_cfg, piece)
    quantized_vid = make_pianoroll_video(qpiece)

    predicted_piece = predict_velocity(model, train_cfg, piece)

    vid = make_pianoroll_video(predicted_piece)

    file_name = f"{model_dir}/pred-{os.path.basename(midi_file.name)}"
    predicted_piece.to_midi().write(file_name)

    return original_vid, quantized_vid, vid, file_name


def main():
    with gr.Blocks(title="MIDI modelling") as demo:
        file = gr.File(file_count="single", label="midi_file")
        with gr.Tab("Both Models"):
            with gr.Row():
                dstart_path = gr.Dropdown(glob.glob("checkpoints/dstart/*.pt"), label="dstart_model")
                velocity_path = gr.Dropdown(glob.glob("checkpoints/velocity/*.pt"), label="velocity_model")
                predict_button = gr.Button("Predict")
            with gr.Row():
                original = gr.Video(label="original_piece")
                quantized = gr.Video(label="quantized_piece")
                out = gr.Video(label="predicted_piece")
            with gr.Row():
                predicted_file = gr.File()

        with gr.Tab("Predict Velocity"):
            with gr.Row():
                velocity_model_path = gr.Dropdown(glob.glob("checkpoints/velocity/*.pt"), label="model")
                velocity_button = gr.Button("Predict")
            with gr.Row():
                vel_original = gr.Video(label="original_piece")
                vel_quantized = gr.Video(label="qantized_piece")
                velocity_out = gr.Video(label="predicted_piece")
            with gr.Row():
                velocity_file = gr.File(label="predicted_midi")

        with gr.Tab("Predict Dstart"):
            with gr.Row():
                dstart_model_path = gr.Dropdown(glob.glob("checkpoints/dstart/*.pt"), label="model")
                dstart_button = gr.Button("Predict")
            with gr.Row():
                dstart_original = gr.Video(label="original_piece")
                dstart_quantized = gr.Video(label="quantized_piece")
                dstart_out = gr.Video(label="predicted_piece")
            with gr.Row():
                dstart_file = gr.File(label="predicted_midi")

        velocity_button.click(
            run_velocity_app,
            inputs=[file, velocity_model_path],
            outputs=[vel_original, vel_quantized, velocity_out, velocity_file],
        )
        dstart_button.click(
            run_dstart_app,
            inputs=[file, dstart_model_path],
            outputs=[dstart_original, dstart_quantized, dstart_out, dstart_file],
        )
        predict_button.click(
            run_predict_app,
            inputs=[file, dstart_path, velocity_path],
            outputs=[original, quantized, out, predicted_file],
        )

    demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
