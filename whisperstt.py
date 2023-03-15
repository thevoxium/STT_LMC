import whisper

model = whisper.load_model("tiny")

result = model.transcribe("audiohin.mp3", language="hi")

print(result["text"])