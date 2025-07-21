import os
import librosa
import soundfile as sf
import shutil


def stretch_audio_to_bpm(audio_path, target_bpm, output_path):
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    current_bpm = librosa.beat.tempo(y=librosa.to_mono(y), sr=sr)[0]
    rate = target_bpm / current_bpm

    print(f"[DEBUG] BPM: {current_bpm:.2f} → {target_bpm}, rate={rate:.4f}")

    y_stretched = librosa.effects.time_stretch(librosa.to_mono(y), rate)
    output_file = os.path.join(output_path, f"{os.path.basename(audio_path).replace('.wav', '')}_stretched.wav")
    sf.write(output_file, y_stretched, sr, subtype="PCM_16")
    return output_file



def pitch_shift_audio(audio_path, semitones, output_path):
    if semitones == 0:
        print("🧊 Pitch shift = 0 → просто копіюю файл")
        output_file = os.path.join(output_path,
                                   os.path.basename(audio_path).replace(".wav", "") + f"_pitch+0.wav")
        shutil.copy(audio_path, output_file)
        return output_file

    print(f"[DEBUG] Pitch shifting {audio_path} by {semitones:+} semitонів")
    y, sr = librosa.load(audio_path, sr=None)
    y_shifted = librosa.effects.pitch_shift(y, sr, semitones)
    output_file = os.path.join(output_path,
                               os.path.basename(audio_path).replace(".wav", "") + f"_pitch{semitones:+}.wav")
    sf.write(output_file, y_shifted, sr)
    return output_file


# Тестовий запуск
if __name__ == "__main__":
    in_file = "../../data/acapellas/lilbaby_wrongturn.wav"
    out_dir = "../../temp/"

    stretched = stretch_audio_to_bpm(in_file, 160, out_dir)
    shifted = pitch_shift_audio(stretched, -1, out_dir)
    print("Done:", shifted)
