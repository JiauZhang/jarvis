import pyttsx3

engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 170)
voices = engine.getProperty('voices')
# 0 for male, 1 for female
engine.setProperty('voice', voices[1].id)

def speak(text):
    print('jarvis:', text)
    engine.say(text)
    engine.runAndWait()
