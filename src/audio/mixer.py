import librosa
import soundfile as sf
import numpy as np
import os

def rms_normalize(y, target_db=-14.0):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_db / 20) / (rms + 1e-9)
    return y * scalar

def estimate_volume_ratio(acapella, beat):
    aca_rms = np.sqrt(np.mean(acapella ** 2))
    beat_rms = np.sqrt(np.mean(beat ** 2))
    if beat_rms == 0:
        return 0.5
    ratio = beat_rms / (aca_rms + 1e-9)
    return min(max(0.65, ratio * 0.85), 0.95)

def mix_multiple_acapellas(beat_path, acapella_paths, output_path):
    print("🎧 Завантажую біт:", os.path.basename(beat_path))
    beat, sr = librosa.load(beat_path, sr=None)
    beat = rms_normalize(beat)

    total_aca = np.zeros_like(beat)
    current_sample = 0

    for i, aca_path in enumerate(acapella_paths):
        print(f"🎙 Додаю акапелу {i+1}/{len(acapella_paths)}: {os.path.basename(aca_path)}")
        aca, _ = librosa.load(aca_path, sr=sr)
        aca = rms_normalize(aca)

        end_sample = current_sample + len(aca)
        if end_sample > len(total_aca):
            end_sample = len(total_aca)
            aca = aca[:end_sample - current_sample]

        total_aca[current_sample:end_sample] += aca
        current_sample = end_sample

    volume_ratio = estimate_volume_ratio(total_aca, beat)
    print(f"[DEBUG] volume_ratio = {volume_ratio:.2f}")

    mixed = (beat * volume_ratio) + (total_aca * (1 - volume_ratio))
    max_val = np.max(np.abs(mixed))
    if max_val > 1:
        mixed = mixed / max_val

    filename = f"converted_beat__{'_'.join([os.path.splitext(os.path.basename(a))[0] for a in acapella_paths])}.wav"
    out_file = os.path.join(output_path, filename)
    sf.write(out_file, mixed, sr, subtype="PCM_16")

    print(f"✅ Мікс збережено: {filename}")
    return out_file, len(mixed) / sr
