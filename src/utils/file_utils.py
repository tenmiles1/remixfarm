import re

def parse_filename(filename):
    """
    Наприклад: '140 bpm F#min ...' → {'bpm': 140, 'key': 'F#min'}
    """
    bpm_match = re.search(r"(\d{2,3})\s?bpm", filename.lower())  # дозволяє пробіл між цифрою і bpm
    key_match = re.search(r"([A-G]#?)(maj|min)", filename, re.IGNORECASE)

    bpm = int(bpm_match.group(1)) if bpm_match else None
    key = f"{key_match.group(1).upper()}{key_match.group(2).lower()}" if key_match else None

    return {
        "bpm": bpm,
        "key": key
    }
