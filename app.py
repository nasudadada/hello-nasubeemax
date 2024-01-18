import os
import streamlit as st
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def transcribe_audio_to_text(audio_bytes):
    with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()
        with open(temp_file.name, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )
        print(response)
    return response


def main():
    st.title("Voice to Text Transcription")

    audio_bytes = audio_recorder(pause_threshold=30)

    # Convert audio to text using OpenAI Whisper API
    if audio_bytes:
        transcript = transcribe_audio_to_text(audio_bytes)
        st.write("Transcribed Text:", transcript)


if __name__ == "__main__":
    main()
