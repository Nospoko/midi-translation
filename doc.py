import os
import glob
import random

import streamlit as st


def set_random(idx: list[int], length: int):
    idx[0] = random.randint(0, length)


def main():
    st.markdown(
        """
# MIDI Velocity Prediction with Transformer Model

### Introduction
MIDI velocity is a crucial element in music dynamics, determining the force with which a note is played, 
which profoundly influences the emotional quality of music. 

Our Transformer-based model aims to decode 
this nuanced aspect of musical expression, unraveling the hidden patterns 
within quantized MIDI data.

### Model Overview
    
The Transformer model is ideal for this task because it excels at capturing complex dependencies in sequential 
data, making it well-suited for predicting MIDI velocities accurately.

This model's suitability arises from its self-attention mechanism,
which enables it to weigh the importance of different parts of the input sequence,
regardless of their temporal order.
In the context of MIDI data, this means that the Transformer can effectively learn
and leverage complex relationships between musical notes, their timing,
and how these factors influence the resulting velocity.

Strong prediction results signify the model's proficiency in extracting vital
features and comprehending intricate relationships. 
Its accurate encoding of quantized MIDI data and precise velocity predictions
mark a significant stride toward the realm of emotionally resonant AI music generation.
### Data Preprocessing
#### MIDI data
MIDI data describes notes by 5 features:
   1. Pitch - Represented as a number between 0 and 127 (or 21 to 108 for piano keys, reflecting the 
   standard 88-key keyboard).
   2. Start - Indicates the moment a key is pressed, measured in seconds.
   3. End - Marks the second when the key is released.
   4. Duration - calculated as the time elapsed between the key's press and release.
   5. Velocity - ranging from 0 to 128, indicating the intensity of the key press.

#### Quantization
To achieve consistent quantization regardless of tempo variations and piece duration, 
we first engineered a more suitable representation of the notes:

1. Pitch - same as above.
2. Dstart - time elapsed after start of previous note.
3. Duration - same as above,
4. Velocity - same as above.

We extracted 128-note samples which we quantized using 3 bins for dstart, 3 for duration and 3 for velocity.
Pitch information remained the same.

Here are **bin edges** we used:
```
dstart:
  - 0.0
  - 0.048177
  - 0.5
duration:
  - 0.0
  - 0.145833
  - 0.450000
velocity:
  - 0.0
  - 57.0
  - 74.0
  - 128.0
```
#### Quantization Samples

        """
    )
    pieces_dir = "documentation/files/pieces"
    samples_dir = "documentation/files/samples"

    q_cols = st.columns(2)
    with q_cols[0]:
        st.markdown("**Real**")
        st.image(f"{samples_dir}/5004-real.png")
        st.audio(f"{samples_dir}/5004-real.mp3")
    with q_cols[1]:
        st.markdown("**Quantized**")
        st.image(f"{samples_dir}/5004-quantized.png")
        st.audio(f"{samples_dir}/5004-quantized.mp3")

    st.markdown(
        """
        ### Model Architecture
A transformer built as described in [Attention is all you need](https://arxiv.org/abs/1706.03762) paper was used.
The important hyperparameters:
- Number of layers in encoder and decoder: **6**
- Nuber of heads in attention layers: **8**
- Dimension of encoder and decoder outputs: **512**
- Dimension of a hidden layer of position-wise fast-forward network from each layer of encoder and decoder: **2048**


### Training and Evaluation
#### Data
   The model was trained on ~200 hours of musical data from 
   [roszcz/maestro-v1](https://huggingface.co/datasets/roszcz/maestro-v1) dataset containing 1276 pieces of 
   classical music performed during piano competition. 
#### Hardware and schedule
   Training on (very old) Nvidia GeForce GTX 960M with 4096 MiB of memory for 5 epochs (2723 steps) took only 7,5 hours.
   Each step took ~6 seconds.
#### Optimizer
Optimizer and learning rate were used as described in
[Attention is all you need](https://arxiv.org/abs/1706.03762) paper:
- Adam optimizer with *β1 = 0.9, β2 = 0.98* and *ϵ = 10−9*.
- The learning rate varied over the course of training, according to the formula:
*lrate = d_model^(-0.5) \* min(step_num^(−0.5), step_num \* warmup_steps^(−1.5))*

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 3000. 
#### Results
The model reaches **2.57 loss** and **5.13 average distance** between prediction and real value.
In contrast - untrained model has a **4.9 loss** and **30.7 average distance**
### Demonstration
#### Random sample
Click below to plot new random sample.
        """
    )
    samples = glob.glob(f"{samples_dir}/*real*.mp3")
    # Using a list, so I will be able to change the value inside the function
    idx = [0]
    st.button("Random", on_click=set_random(idx, length=len(samples) - 1))
    sample_idx = os.path.basename(samples[idx[0]]).split('-')[0]
    sample_cols = st.columns(4)
    with sample_cols[0]:
        st.markdown("**Original**")
        st.image(f"{samples_dir}/{sample_idx}-real.png")
        st.audio(f"{samples_dir}/{sample_idx}-real.mp3")
    with sample_cols[1]:
        st.markdown("**Quantized**")
        st.image(f"{samples_dir}/{sample_idx}-quantized.png")
        st.audio(f"{samples_dir}/{sample_idx}-quantized.mp3")
    with sample_cols[2]:
        st.markdown("**Q. velocity**")
        st.image(f"{samples_dir}/{sample_idx}-qv.png")
        st.audio(f"{samples_dir}/{sample_idx}-qv.mp3")
    with sample_cols[3]:
        st.markdown("**Predicted**")
        st.image(f"{samples_dir}/{sample_idx}-predicted.png")
        st.audio(f"{samples_dir}/{sample_idx}-predicted.mp3")
    st.markdown(
        """
#### Pieces with predicted velocity
You can choose which piece you would like to listen to and compare it's original version with the one predicted by our
model.
        """
    )

    pieces = glob.glob(f"{pieces_dir}/*-pred.mp3")
    piece_to_plot = st.selectbox(
        label="Piece to plot",
        options=[os.path.basename(piece).split(".")[0][:-5] for piece in pieces],
    )
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Original**")
        st.image(f"documentation/files/pieces/{piece_to_plot}.png")
        st.audio(f"documentation/files/pieces/{piece_to_plot}.mp3")
    with cols[1]:
        st.markdown("**Predicted velocity**")
        st.image(f"documentation/files/pieces/{piece_to_plot}-pred.png")
        st.audio(f"documentation/files/pieces/{piece_to_plot}-pred.mp3")


if __name__ == "__main__":
    main()
