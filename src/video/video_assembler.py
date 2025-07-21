import os

from moviepy.editor import VideoFileClip, concatenate_videoclips
import random

def safe_subclip(path, start, end):
    video = VideoFileClip(path)
    duration = video.duration
    if start >= duration:
        print(f'⛔️ Start {start} >= video duration {duration}. Пропуск.')
        return None
    if end > duration:
        print(f'⚠️ End {end} > video duration {duration}. Підрізаємо.')
        end = duration
    return video.subclip(start, end)

def assemble_video_with_neutral(track_duration, neutral_segments, mouth_moving_chunks):
    if len(neutral_segments) > 1:
        random.shuffle(neutral_segments)
        print("Order after shuffle:", [os.path.basename(seg[0]) for seg in neutral_segments])
    cuts = []
    prev_end = 0
    for seg in neutral_segments:
        path, start, end = seg
        if start > prev_end:
            cuts.append((prev_end, start, 'main'))
        cuts.append((start, end, 'neutral', path))
        prev_end = end
    if prev_end < track_duration:
        cuts.append((prev_end, track_duration, 'main'))

    clips = []
    last_idx = -1
    for cut in cuts:
        if cut[2] == 'main':
            left = cut[1] - cut[0]
            while left > 0:
                candidates = [i for i in range(len(mouth_moving_chunks)) if i != last_idx]
                idx = random.choice(candidates) if candidates else last_idx
                vclip = VideoFileClip(mouth_moving_chunks[idx])
                use_len = min(vclip.duration, left)
                clips.append(vclip.subclip(0, use_len))
                left -= use_len
                last_idx = idx
        else:
            path, start, end = cut[3], cut[0], cut[1]
            clip = safe_subclip(path, start, end)
            if clip:
                clips.append(clip)
    final_clip = concatenate_videoclips(clips, method="compose")
    print(f'🎬 Зібрано кліп: {final_clip.duration:.2f} сек (трек: {track_duration} сек)')

    # Експортуємо відео тут!
    final_clip.write_videofile("output/final_mix_video.mp4", fps=30, codec="libx264", audio_codec="aac")

    # === ДУЖЕ ВАЖЛИВО ===
    for clip in clips:
        clip.close()
    final_clip.close()
    # ====================

    # Якщо хочеш — можна return None, бо відео вже збережено
