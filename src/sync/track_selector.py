import os
from src.utils.file_utils import parse_filename


def get_audio_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".wav") or f.endswith(".mp3")]


def parse_key(key_str):
    """Конвертує ноти в MIDI номера для порівняння тональностей"""
    if not key_str:
        return -1
    key_str = key_str.upper().replace("MIN", "").replace("MAJ", "")
    key_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
               'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    return key_map.get(key_str, -1)


def find_best_matches(acapella_folder, beat_folder, bpm_tolerance=5, max_semitone_shift=2):
    acapellas = get_audio_files(acapella_folder)
    beats = get_audio_files(beat_folder)

    matches = []

    for aca_file in acapellas:
        aca_path = os.path.join(acapella_folder, aca_file)
        aca_meta = parse_filename(aca_file)
        aca_bpm = aca_meta.get("bpm")
        aca_key = aca_meta.get("key")
        aca_key_val = parse_key(aca_key)

        if aca_bpm is None or aca_key_val == -1:
            continue  # скіпаємо якщо не зчиталось

        for beat_file in beats:
            beat_path = os.path.join(beat_folder, beat_file)
            beat_meta = parse_filename(beat_file)
            beat_bpm = beat_meta.get("bpm")
            beat_key = beat_meta.get("key")
            beat_key_val = parse_key(beat_key)

            if beat_bpm is None or beat_key_val == -1:
                continue

            bpm_diff = abs(aca_bpm - beat_bpm)
            key_diff = abs(aca_key_val - beat_key_val)

            if bpm_diff <= bpm_tolerance and key_diff <= max_semitone_shift:
                matches.append({
                    "acapella": aca_path,
                    "beat": beat_path,
                    "bpm_diff": bpm_diff,
                    "key_diff": key_diff
                })

    matches.sort(key=lambda x: (x["key_diff"], x["bpm_diff"]))
    return matches


# Тест (можеш запустити прямо)
if __name__ == "__main__":
    matches = find_best_matches("../../acapellas", "../../beats")
    for m in matches:
        print(f"✅ Match: {os.path.basename(m['acapella'])} + {os.path.basename(m['beat'])} | ΔBPM: {m['bpm_diff']} ΔKey: {m['key_diff']}")
