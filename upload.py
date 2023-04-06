from __future__ import division
from google.oauth2 import service_account
import streamlit as st
import re
import sys
from google.cloud import speech_v1p1beta1 as speech
import io
from six.moves import queue
from six import binary_type
import os
from pydub import AudioSegment
from google.cloud import translate_v2 as translate
from tempfile import NamedTemporaryFile

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

st.title("Last Mile Care")


RATE = 48000
language_code = "hi-IN"


translate_client = translate.Client(credentials=credentials)


def main():
    language_code = "hi-IN"
    client = speech.SpeechClient(credentials=credentials)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    uploaded_file = st.file_uploader("Choose an audio file (.mp3)", type=["mp3"])
    if uploaded_file is not None:
        with NamedTemporaryFile(suffix=".flac") as temp:
            mp3_audio = AudioSegment.from_file(io.BytesIO(uploaded_file.read()), format="mp3")
            mp3_audio.export(temp.name, format="flac")

            
            st.markdown('<div style="color:#23AB35">Transcribing...</div>', unsafe_allow_html=True)

            with io.open(temp.name, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            response = client.recognize(config=config, audio=audio)

            for result in response.results:

                transcript = result.alternatives[0].transcript
                st.write(transcript)

                st.markdown('<div style="color:#23AB35">Translating...</div>', unsafe_allow_html=True)

                result_translated = translate_client.translate(transcript)["translatedText"]
                st.write(result_translated)


if __name__ == "__main__":
    main()
