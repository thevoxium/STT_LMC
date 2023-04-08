import argparse
import os 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/anshul/keen-honor-287510-5084f26416a6.json"

def transcribe_file(speech_file):
    from google.cloud import speech
    import io

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        language_code="hi-IN",
    )
    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))
        with open('hindi_sample.txt', 'w') as f:
            f.write(result.alternatives[0].transcript)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="File or GCS path for audio file to be recognized")
    args = parser.parse_args()
    transcribe_file(args.path)