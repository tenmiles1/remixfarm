import librosa
import numpy as np

def detect_bpm(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    tempo, _ = librosa.beat.beat_track(y=librosa.to_mono(y), sr=sr)
    return round(tempo)

def detect_key(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    chroma = librosa.feature.chroma_cens(y=librosa.to_mono(y), sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    key_index = np.argmax(chroma_avg)

    major_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return major_keys[key_index]

