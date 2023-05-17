from __future__ import division
from google.oauth2 import service_account
import streamlit as st
import re
import sys
from google.cloud import speech
import pyaudio
from six.moves import queue
from six import binary_type
import os
from google.cloud import translate_v2 as translate
import mysql.connector
from transformers import T5ForConditionalGeneration, T5Tokenizer


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

st.title("Speech-to-Text with Google's STT API")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

RATE = 16000
CHUNK = int(RATE / 10)
language_code = "hi-IN"


translate_client = translate.Client(credentials=credentials)


def answer_question(question):
    inputs = tokenizer.encode("question: " + question, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def connect_to_database():
    return mysql.connector.connect(
        host=st.secrets["mysql_host"],
        user=st.secrets["mysql_user"],
        password=st.secrets["mysql_password"],
        database="lastmilecare",
    )

def insert_transcription(transcript, translated_text, feedback=90):
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

class MicrophoneStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,            
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self
    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()
    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue
    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

def listen_print_loop(responses):
    num_chars_printed = 0

    placeholder = st.empty()

    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        overwrite_chars = " " * (num_chars_printed - len(transcript))
        if not result.is_final:
            placeholder.text(transcript+overwrite_chars)
            num_chars_printed = len(transcript)
        else:
            placeholder.empty()
            st.write(transcript + overwrite_chars)
            req_text = transcript + overwrite_chars

            if isinstance(req_text, binary_type):
                req_text = req_text.decode("utf-8")
            
            result = translate_client.translate(req_text)["translatedText"]
            #st.markdown('<div style="color:#23AB35">{}</div>'.format(result), unsafe_allow_html=True)
            
            answer = answer_question(result)
            st.write(f"Answer: {answer}")

            transcription_id = insert_transcription(transcript+overwrite_chars, result, 92)
            st.write(f"Inserted transcription with ID: {transcription_id}")

            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                st.write("Exiting..")
                break
            num_chars_printed = 0

def main():
    language_code = "hi-IN"  
    client = speech.SpeechClient(credentials=credentials)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True, single_utterance = False
    )


    is_recording = False
    st.text("Click 'Start Recording'")
    start_button = st.button("Start Recording")
    stop_button = st.button("Stop Recording and Clear Text")

    with st.spinner("Starting the recorder..."):
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )
            if start_button:
                is_recording = True
            if stop_button:
                is_recording = False

            if is_recording:
                responses = client.streaming_recognize(streaming_config, requests)
                listen_print_loop(responses)
            else:
                st.text("Click 'Start Recording' to start transcribing audio from your microphone")

if __name__ == "__main__":
    main()
