import os
import glob

import torch
import gradio as gr
import fortepyan as ff
from omegaconf import OmegaConf

from dashboard.gradio.tools import load_model, predict_dstart, quantize_piece, predict_velocity, make_pianoroll_video

DESCRIPTION = """
<h1>🎵 Modelling dynamic expression in music. 🎶</h1>
<h3>AI Pianist</h3>
<p>This interactive application uses an AI model to generate music sequences based on quantized midi files.
You can upload your midi file and it will be quantized, passed to the model and it will play it with expression.</p>
<div style="display: flex; justify-content: space-between;">
    <div style="width: 45%; margin-right: 5%;">
        <h2>Features:</h2>
        <ul>
            <li>🎹 Upload your midi file with piano performance.</li>
            <li>🎼 Select the target (velocity, dstart or both).</li>
            <li>▶️ Click the predict button and await the result!</li>
        </ul>
    </div>
    <div style="width: 45%; margin-left: 5%;">
        <h2>Outputs:</h2>
        <p>The app outputs the following:</p>
        <ul>
            <li>🎧 The audio and pianoroll of the generated song.</li>
            <li>📁 A MIDI file of the song.</li>
        </ul>
    </div>
</div>
"""


def run_sequential_pred_app(
    midi_file,
    velocity_model_path: str,
    dstart_model_path: str,
    progress=gr.Progress(track_tqdm=True),
):
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

    pred_velocity = predict_velocity(velocity_model, velocity_cfg, piece)
    qpiece.df = qpiece.df.head(len(pred_velocity.df))
    qpiece.df["velocity"] = pred_velocity.df["velocity"]
    pred_velocity_vid = make_pianoroll_video(qpiece)

    pred_dstart = predict_dstart(dstart_model, dstart_cfg, pred_velocity)

    vid = make_pianoroll_video(pred_dstart)

    file_name = f"tmp/pred-{os.path.basename(midi_file.name)}"
    pred_dstart.to_midi().write(file_name)

    return original_vid, quantized_vid, pred_velocity_vid, vid, file_name


def run_predict_app(midi_file, velocity_model_path: str, dstart_model_path: str, progress=gr.Progress(track_tqdm=True)):
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
    # layout
    with gr.Blocks(title="MIDI modelling") as demo:
        gr.HTML(DESCRIPTION)
        file = gr.File(file_count="single", label="midi_file")
        with gr.Tab("Both Models"):
            with gr.Row():
                velocity_path = gr.Dropdown(glob.glob("checkpoints/velocity/*.pt"), label="velocity_model")
                dstart_path = gr.Dropdown(glob.glob("checkpoints/dstart/*.pt"), label="dstart_model")
                predict_button = gr.Button("Predict")
            with gr.Row():
                original = gr.Video(label="original_piece")
                quantized = gr.Video(label="quantized_piece")
                out = gr.Video(label="predicted_piece")
            with gr.Row():
                predicted_file = gr.File()

        with gr.Tab("Sequential predictions"):
            with gr.Row():
                seq_velocity_path = gr.Dropdown(glob.glob("checkpoints/velocity/*.pt"), label="velocity_model")
                seq_dstart_path = gr.Dropdown(glob.glob("checkpoints/dstart/*128v*.pt"), label="dstart_model")
                seq_predict_button = gr.Button("Predict")
            with gr.Row():
                seq_original = gr.Video(label="original_piece")
                seq_quantized = gr.Video(label="quantized_piece")
                seq_pred_velocity = gr.Video(label="predicted_velocity")
                seq_out = gr.Video(label="predicted_piece")
            with gr.Row():
                seq_predicted_file = gr.File()

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

        # button clicks
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
            inputs=[file, velocity_path, dstart_path],
            outputs=[original, quantized, out, predicted_file],
        )
        seq_predict_button.click(
            run_sequential_pred_app,
            inputs=[file, seq_velocity_path, seq_dstart_path],
            outputs=[seq_original, seq_quantized, seq_pred_velocity, seq_out, seq_predicted_file],
        )

    demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
