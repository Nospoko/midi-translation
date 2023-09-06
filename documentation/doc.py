import streamlit as st


def main():
    st.markdown(
        "## MIDI Transformer \n"
        "### Description \n"
        "### Results"
    )
    cols = st.columns(3)
    with cols[0]:
        st.image("documentation/files/2009-0-MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_07_WAV.png")
        st.audio("documentation/files/2009-0-MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_07_WAV.mp3")
    with cols[1]:
        st.image("documentation/files/2009-0-target-3-3-3-MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_07_WAV.png")
        st.audio("documentation/files/2009-0-target-3-3-3-MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_07_WAV.mp3")

if __name__ == '__main__':
    main()