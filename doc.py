import os
import glob
import random

import streamlit as st


def set_random(idx: int, length: int):
    idx = random.randint(0, length)


def main():
    st.markdown(
        "## MIDI Transformer: Predicting Velocity from Quantized MIDI Data\n"
        "### Description \n"
        "In this presentation, we unveil the results of our transformative project at the intersection of "
        "AI and music. Our mission: Predicting velocity from quantized MIDI data."
        "### Results"
    )

    pieces = glob.glob("documentation/files/pieces/*(!-pred).mp3")
    print(pieces)
    samples = glob.glob("documentation/files/samples/real*.mp3")
    quantization_samples = glob.glob("documentation/files/quantization_samples/*.mp3")
    idx = 0
    st.button("Random", on_click=set_random(idx, len(samples)))
    piece_to_plot = st.selectbox(
        label="Piece to plot",
        options=[os.path.basename(piece).split(".")[0] for piece in pieces],
    )
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Original**")
        st.image(f"documents/files/pieces/{piece_to_plot}.png")
        st.audio(f"documents/files/pieces/{piece_to_plot}.mp3")
    with cols[1]:
        st.markdown("**Predicted velocity")
        st.image(f"documents/files/pieces/{piece_to_plot}-pred.png")
        st.audio(f"documents/files/pieces/{piece_to_plot}-pred.mp3")
    print(piece_to_plot)


if __name__ == "__main__":
    main()
