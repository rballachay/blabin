import speech_recognition as sr

r = sr.Recognizer()
r.energy_threshold = 4000

with sr.AudioFile('data/testing-audio.wav') as source:  # open the audio file for reading
    audio = r.listen(source, phrase_time_limit=4)

    text = r.recognize_faster_whisper(audio, model='large-v3')
    print(text)
