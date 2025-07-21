import os
import random
from moviepy.editor import VideoFileClip

def split_video(video_path, chunk_length_range=(5, 10), output_folder="temp/video_chunks"):
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    clip = VideoFileClip(video_path)
    duration = clip.duration

    chunks = []
    t = 0
    idx = 0

    while t < duration - 5:
        chunk_len = random.uniform(*chunk_length_range)
        end_time = min(t + chunk_len, duration)
        chunk = clip.subclip(t, end_time)

        out_path = os.path.join(output_folder, f"{base_name}_part{idx}.mp4")
        chunk.write_videofile(out_path, codec="libx264", audio=False, verbose=False, logger=None)
        print(f"  ✂️ Чанк {out_path} [{t:.1f}–{end_time:.1f}] сек")
        chunks.append(out_path)
        t += chunk_len
        idx += 1

    clip.close()
    return chunks
