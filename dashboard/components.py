import re
import json
import uuid
import base64
import hashlib

import pandas as pd
import streamlit as st
from datasets import load_dataset
from fortepyan import MidiFile, MidiPiece


def piece_selector(dataset_name: str) -> tuple[MidiPiece, str]:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        midi_file = MidiFile(uploaded_file)
        piece = midi_file.piece
        piece.source["path"] = "file uploaded with streamlit"

        # Use file md5 instead of dataset name
        file_hash = hashlib.md5()
        uploaded_file.seek(0)
        file_hash.update(uploaded_file.read())
        piece_descriptor = file_hash.hexdigest()
    else:
        st.write("Or use a dataset")
        dataset_name = st.text_input(label="dataset", value=dataset_name)
        split = st.text_input(label="split", value="test")

        # Test/77 is Chopin "Etude Op. 10 No. 12"
        record_id = st.number_input(label="record id", value=77)
        hf_dataset = load_dataset(dataset_name, split=split)

        # Select one full piece
        record = hf_dataset[record_id]
        piece = MidiPiece.from_huggingface(record)
        piece_descriptor = f"{dataset_name}-{split}-{record_id}"

    return piece, piece_descriptor


def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if isinstance(object_to_download, bytes):
        pass

    elif isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub(r"\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    a_html = f"""
    <a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">
        {button_text}
    </a>
    <br></br>
    """
    button_html = custom_css + a_html

    return button_html
