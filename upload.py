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
import mysql.connector

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

st.title("Last Mile Care")

option = st.selectbox(
    'Select your device microphone input type',
    ('Laptop (Single Mic)', 'Laptop (Dual Mic)', 'Mobile (Single Mic)', 'Mobile (Dual Mic)'))


RATE = 48000
language_code = "hi-IN"

audio_channel_count = 1

if (option == 'Laptop (Single Mic)' or option == 'Mobile (Single Mic)'):
    audio_channel_count = 1
else:
    audio_channel_count = 2

translate_client = translate.Client(credentials=credentials)


def connect_to_database():
    return mysql.connector.connect(
        host=st.secrets["mysql_host"],
        user=st.secrets["mysql_user"],
        password=st.secrets["mysql_password"],
        database="lastmilecare",
    )

def insert_transcription(transcript, translated_text, feedback):
    connection = connect_to_database()
    cursor = connection.cursor()
    
    query = """
        INSERT INTO results (transcript, translated_text, feedback)
        VALUES (%s, %s, %s)
    """
    
    cursor.execute(query, (transcript, translated_text, feedback))
    connection.commit()
    
    transcription_id = cursor.lastrowid
    
    cursor.close()
    connection.close()
    
    return transcription_id


def main():
    language_code = "hi-IN"
    client = speech.SpeechClient(credentials=credentials)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        language_code=language_code,
        audio_channel_count = audio_channel_count
    )

    uploaded_file = st.file_uploader("Choose an audio file (.mp3, .ogg)", type=["mp3", "ogg"])

    if uploaded_file is not None:
        file_format = uploaded_file.type.split("/")[-1]
        with NamedTemporaryFile(suffix=".flac") as temp:
            mp3_audio = AudioSegment.from_file(io.BytesIO(uploaded_file.read()), format=file_format)
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

                feedback = st.slider("Feedback", min_value = 0, max_value = 100, step = 1)

                database_update_btn = st.button("Upload Results to Database")
                if (database_update_btn):
                    transcription_id = insert_transcription(transcript, result_translated, feedback)
                    st.write(f"Inserted transcription with ID: {transcription_id}")


if __name__ == "__main__":
    main()
