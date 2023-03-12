from jarvis.microphone import Microphone, Recognizer
from jarvis.tts import speak
import numpy as np
import whisper

model = whisper.load_model("tiny.en").to('cpu')

r = Recognizer()
with Microphone() as source:
    speak("I am Jarvis your personal voice assistant, please tell me how could I help you?")
    speak("I'm Listening...")
    audio = r.listen(source)
    raw_data = audio.get_raw_data(convert_rate=16000)
    norm_data = np.frombuffer(raw_data, np.int16).flatten().astype(np.float32) / 32768.0
    speak('transcribing...')
    result = model.transcribe(norm_data)
    speak(result['text'])
