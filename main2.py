import os
import sys
import json
import glob
import numpy as np
import pytesseract
from PIL import Image
import cv2
from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips
from src.video.video_assembler import safe_subclip
from src.utils.temp_tools import clear_temp_chunks, temp_chunks_exist
import subprocess
import random

# Вкажи шлях до tesseract якщо не прописано в PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

TRACK_PATH = "data/beats/1707.wav"
OUTPUT_PATH = "output/final_mix_video.mp4"
VIDEO_CHUNKS_DIR = "temp/video_chunks"
FRAME_SIZE = (1920, 1080)
FPS = 30

def safe_close(clip):
    try:
        if hasattr(clip, 'close'):
            clip.close()
    except Exception:
        pass
    try:
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
    except Exception:
        pass
    try:
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close_proc()
    except Exception:
        pass
    import gc
    gc.collect()

def remux_chunks(input_dir):
    import glob
    import subprocess
    import os
    for f in glob.glob(f"{input_dir}/**/*.mp4", recursive=True):
        temp_out = f.replace(".mp4", "_remux.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", f, "-c:v", "copy", "-an", temp_out
        ]
        print(f"Remux: {f} -> {temp_out}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        os.replace(temp_out, f)
    print("✅ Всі chunk-и remuxed!")

def reencode_video_chunks(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                inp = os.path.join(root, file)
                outp = os.path.join(output_dir, file)
                if os.path.exists(outp):
                    continue
                cmd = [
                    "ffmpeg", "-y", "-i", inp,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                    "-c:a", "aac", "-strict", "experimental", outp
                ]
                print(f"Перекодовуємо: {inp} -> {outp}")
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"✅ Всі чанки перекодовано у {output_dir}")

def get_video_fps(video_path):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    safe_close(clip)
    return fps

def cut_video_by_frames(in_file, out_file, start_frame, end_frame, fps):
    import ffmpeg
    duration = (end_frame - start_frame) / fps
    start_time = start_frame / fps
    (
        ffmpeg
        .input(in_file, ss=start_time)
        .output(out_file, t=duration, vcodec='libx264', acodec='aac', loglevel='error')
        .overwrite_output()
        .run()
    )

def slice_video_chunks_frameaccurate(video_path, out_dir, min_scene_len=2, max_scene_len=6, threshold=30.0, fps=60):
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector

    artist_name = os.path.basename(video_path).rsplit('.', 1)[0]
    artist_subdir = os.path.join(out_dir, artist_name)
    os.makedirs(artist_subdir, exist_ok=True)

    print(f"==> SCENEDETECT: threshold={threshold}, min_scene_len={min_scene_len}, fps={fps}")
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len * fps)))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode=video_manager.get_base_timecode())
    video_manager.release()

    print(f"==> Знайдено сцен: {len(scene_list)}")
    chunk_paths = []
    chunk_idx = 1
    for (start, end) in scene_list:
        start_frame = start.get_frames()
        end_frame = end.get_frames()
        while end_frame - start_frame > max_scene_len * fps:
            curr_end = start_frame + int(max_scene_len * fps)
            out_path = os.path.join(artist_subdir, f"{artist_name}_part{chunk_idx}.mp4")
            cut_video_by_frames(video_path, out_path, start_frame, curr_end, fps)
            chunk_paths.append(out_path)
            print(f"==> Ріжу: frame {start_frame}-{curr_end-1}")
            start_frame = curr_end
            chunk_idx += 1
        if end_frame - start_frame >= 1 * fps:
            out_path = os.path.join(artist_subdir, f"{artist_name}_part{chunk_idx}.mp4")
            cut_video_by_frames(video_path, out_path, start_frame, end_frame, fps)
            chunk_paths.append(out_path)
            print(f"==> Ріжу: frame {start_frame}-{end_frame-1}")
            chunk_idx += 1
    print(f"✂️ Нарізано {chunk_idx-1} чанків у {artist_subdir}")

    random.shuffle(chunk_paths)
    return chunk_paths

def resize_and_fps(clip, size=FRAME_SIZE, fps=FPS):
    if clip.size != size:
        clip = clip.resize(size)
    if abs(clip.fps - fps) > 0.5:
        clip = clip.set_fps(fps)
    return clip

def pick_neutral_segments_auto(neutral_chunks, total_len):
    import random
    random.shuffle(neutral_chunks)
    segments = []
    left = total_len
    for path in neutral_chunks:
        clip = VideoFileClip(path)
        duration = clip.duration
        safe_close(clip)
        use_len = min(left, duration)
        segments.append((path, 0, use_len))
        left -= use_len
        if left <= 0:
            break
    if left > 0:
        print(f"⚠️ Увага: нейтральних чанків не вистачає для {total_len} сек. Буде тільки {total_len - left} сек.")
    return segments

def fill_main_blocks(track_duration, neutral_segments, mouth_moving_chunks):
    import random
    from moviepy.video.fx.all import colorx, crop, resize

    random.shuffle(mouth_moving_chunks)
    random.shuffle(neutral_segments)

    cuts = []
    prev_end = 0
    for seg in sorted(neutral_segments, key=lambda x: x[1]):
        path, start, end = seg
        if start > prev_end:
            cuts.append((prev_end, start, 'main'))
        cuts.append((start, end, 'neutral', path))
        prev_end = end
    if prev_end < track_duration:
        cuts.append((prev_end, track_duration, 'main'))

    clips, chunk_paths_for_xml = [], []
    last_video_source = None
    total_time = 0

    print(f"🎯 START fill_main_blocks: target duration={track_duration:.2f}s")

    for cut in cuts:
        left = cut[1] - cut[0]
        print(f"➡️ Block: {cut} (target {left:.2f}s)")

        if cut[2] == 'main':
            iteration = 0
            while left > 0.05 and iteration < 100:
                iteration += 1
                available = [
                    c for c in mouth_moving_chunks
                    if os.path.basename(os.path.dirname(c)) != last_video_source
                ]

                if not available:
                    print("🔁 Дозволяю повтор того самого кліпу")
                    available = mouth_moving_chunks.copy()

                chunk_path = random.choice(available)
                vclip = VideoFileClip(chunk_path).resize(FRAME_SIZE)
                vclip = vclip.fx(colorx, 1.01 + random.uniform(-0.01, 0.01))
                vclip = vclip.fx(crop, x1=1, y1=1)

                use_len = min(vclip.duration, left)
                if use_len < 0.2:
                    vclip.close()
                    continue
                subclip = vclip.subclip(0, use_len).without_audio()
                clips.append(subclip)
                chunk_paths_for_xml.append(chunk_path)
                left -= use_len
                total_time += use_len
                last_video_source = os.path.basename(os.path.dirname(chunk_path))
                print(f"  ✅ Added REP: {use_len:.2f}s from {os.path.basename(chunk_path)}")
                vclip.close()

        else:
            n_path, n_start, n_end = cut[3], cut[0], cut[1]
            clip = safe_subclip(n_path, n_start, n_end)
            if clip and (n_end - n_start) > 0.2:
                clip = resize_and_fps(clip).without_audio()
                clip = clip.fx(colorx, 1.02).fx(crop, x1=1, y1=1)
                clips.append(clip)
                chunk_paths_for_xml.append(n_path)
                total_time += (n_end - n_start)
                print(f"  🟧 Added NEUTRAL: {(n_end - n_start):.2f}s")

    print(f"✅ DONE fill_main_blocks: total built duration={total_time:.2f}s (expected {track_duration:.2f}s)")
    final_clip = concatenate_videoclips(clips, method="compose").subclip(0, track_duration)
    return final_clip, clips, chunk_paths_for_xml

def has_text_in_chunk(video_path, frames_to_check=5, min_text_len=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frames_to_check, 1)
    found_text = False
    for i in range(frames_to_check):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret: continue
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(pil_img)
        if len(text.strip()) >= min_text_len:
            found_text = True
            break
    cap.release()
    return found_text

def find_artists_and_videos(base_folder="data/videos"):
    artists = []
    for artist_dir in os.listdir(base_folder):
        full_path = os.path.join(base_folder, artist_dir)
        if os.path.isdir(full_path):
            videos = [os.path.basename(f) for f in glob.glob(os.path.join(full_path, "*.mp4"))]
            artists.append({'name': artist_dir, 'videos': videos})
    return artists

def ask_artists_order(artists):
    print("Артисти знайдені:")
    for i, a in enumerate(artists):
        print(f"[{i+1}] {a['name']} ({len(a['videos'])} videos)")
    idxs = input("Введи номери артистів у потрібному порядку (через кому): ").strip().split(",")
    idxs = [int(i)-1 for i in idxs if i.strip().isdigit()]
    return [artists[i] for i in idxs]

def ask_videos_for_artist(artist, base_folder="data/videos"):
    print(f"\n[ARTIST: {artist['name']}]")
    if not artist['videos']:
        print("  Кліпів не знайдено!")
        return []
    print("Доступні кліпи:")
    for i, v in enumerate(artist['videos']):
        print(f" [{i+1}] {v}")
    idxs = input("Введи номери кліпів через кому (Enter — всі): ").strip()
    if not idxs:
        return [os.path.join(base_folder, artist['name'], v) for v in artist['videos']]
    idxs = [int(i)-1 for i in idxs.split(",") if i.strip().isdigit()]
    return [os.path.join(base_folder, artist['name'], artist['videos'][i]) for i in idxs]

def ask_artist_end_times(num, total_duration):
    ends = []
    for i in range(num-1):
        sec = None
        while sec is None:
            inp = input(f"⏳ Вкажи ДО ЯКОЇ М:С трек із артиста {i+1} (наприклад, 1:36): ").strip()
            try:
                sec = parse_timecode(inp)
            except:
                print("❌ Неправильний формат. Введи М:С або секунди.")
        ends.append(sec)
    ends.append(total_duration)
    return ends

if __name__ == "__main__":
    artists_all = find_artists_and_videos("data/videos")
    artists_chosen = ask_artists_order(artists_all)
    artist_video_paths = []
    for art in artists_chosen:
        vids = ask_videos_for_artist(art, "data/videos")
        artist_video_paths.append(vids)

    def parse_timecode(input_str):
        if ":" in input_str:
            minutes, seconds = map(int, input_str.strip().split(":"))
            return minutes * 60 + seconds
        else:
            return float(input_str)

    intro_len = 0
    try:
        intro_input = input("⏳ Вкажи довжину інтро (М:С або секунди, Enter якщо не треба): ").strip()
        if intro_input:
            intro_len = parse_timecode(intro_input)
    except Exception:
        intro_len = 0

    print(f"📋 Інтро: {intro_len:.2f} сек")
    track_audio = AudioFileClip(TRACK_PATH)
    TRACK_DURATION = track_audio.duration
    safe_close(track_audio)

    print(f"🎵 Тривалість треку: {TRACK_DURATION:.2f} сек")
    artist_end_times = ask_artist_end_times(len(artists_chosen), TRACK_DURATION)
    artists_timeline = []
    start = 0
    for i, (art, vids) in enumerate(zip(artists_chosen, artist_video_paths)):
        artists_timeline.append({
            'name': art['name'],
            'videos': vids,
            'start': start,
            'end': artist_end_times[i]
        })
        start = artist_end_times[i]

    print("\n=== Обраний порядок артистів ===")
    for a in artists_timeline:
        print(f"{a['name']}: {a['start']}–{a['end']} сек ({len(a['videos'])} відео)")

    final_blocks, chunk_paths_all = [], []

    for idx, artist in enumerate(artists_timeline):
        print(f"\n=== Обробка артиста: {artist['name']} ({artist['start']}–{artist['end']} сек) ===")
        artist_chunk_dir = os.path.join(VIDEO_CHUNKS_DIR, artist['name'])
        os.makedirs(artist_chunk_dir, exist_ok=True)
        chunk_paths = []
        for video_path in artist['videos']:
            video_base = os.path.splitext(os.path.basename(video_path))[0]
            sub_chunk_dir = os.path.join(artist_chunk_dir, video_base)
            if not os.path.exists(sub_chunk_dir) or not os.listdir(sub_chunk_dir):
                print(f"✂️ Нарізаю {video_path} ...")
                real_fps = get_video_fps(video_path)
                paths = slice_video_chunks_frameaccurate(
                    video_path, artist_chunk_dir,
                    min_scene_len=4, max_scene_len=8, threshold=20.0, fps=real_fps
                )
                chunk_paths.extend(paths)
            else:
                paths = [os.path.join(sub_chunk_dir, f) for f in sorted(os.listdir(sub_chunk_dir)) if
                         f.endswith('.mp4')]
                chunk_paths.extend(paths)
        print(f"Загалом нарізано чанків: {len(chunk_paths)}")

        chunk_info_json = f"temp/chunk_lipsync_info_{artist['name']}.json"
        if not os.path.exists(chunk_info_json):
            print("🟠 Класифікую lipsync для артиста...")
            from src.video.lipsync_classifier import classify_all_chunks

            chunk_info = classify_all_chunks(artist_chunk_dir)
            with open(chunk_info_json, "w") as f:
                json.dump(chunk_info, f, indent=2, ensure_ascii=False)
        else:
            with open(chunk_info_json, "r") as f:
                chunk_info = json.load(f)

        all_mean_movings = [c["mean_moving"] for c in chunk_info if "mean_moving" in c]
        if len(all_mean_movings) > 1:
            threshold = float(np.median(all_mean_movings)) + 0.03
        else:
            threshold = 0.12

        mouth_moving_chunks = [c["path"] for c in chunk_info if c.get("mean_moving", 0) > threshold]
        neutral_chunks = [c["path"] for c in chunk_info if c.get("mean_moving", 0) <= threshold]

        print(f"Порогове значення mean_moving для реп-чанків: {threshold:.4f}")

        block_duration = artist['end'] - artist['start']

        if not mouth_moving_chunks:
            print("❌ Не знайдено реп-чанків (mouth_moving). Перевір lipsync_classifier або чанки.")
            continue

        neutral_segments = []
        if idx == 0 and intro_len > 0:
            print("\n--- Автовибір інтро ---")
            print("Фільтрую інтро чанки без тексту...")
            filtered_neutral_chunks = [p for p in neutral_chunks if not has_text_in_chunk(p)]
            neutral_segments = pick_neutral_segments_auto(filtered_neutral_chunks, intro_len)
            print("Інтро буде з наступних шматків (без тексту):")
            for seg in neutral_segments:
                print(f"{os.path.basename(seg[0])}: {seg[1]} - {seg[2]} сек")

        final_clip, clips, chunk_paths_for_xml = fill_main_blocks(block_duration, neutral_segments, mouth_moving_chunks)
        print(f"✅ fill_main_blocks завершено для {artist['name']}, тривалість: {block_duration:.2f}s")
        if final_clip:
            final_blocks.append(final_clip)
            chunk_paths_all.extend(chunk_paths_for_xml)
            print(f"✅ Блок для {artist['name']} додано у final_blocks")
        else:
            print(f"❌ Блок для {artist['name']} не додано (final_clip = None)")
        for c in clips:
            safe_close(c)

    reencode_video_chunks("temp/video_chunks", "temp/video_chunks_reencoded")
    remux_chunks("temp/video_chunks_reencoded")
    VIDEO_CHUNKS_DIR = "temp/video_chunks_reencoded"

    print("\n🎬 Об'єдную відео-блоки артистів у фінальний кліп...")

    import shutil

    def export_and_concat_blocks(final_blocks, output_path, track_audio_path=None, fps=30):
        temp_dir = "output/temp_blocks"
        os.makedirs(temp_dir, exist_ok=True)
        temp_files = []
        print("\n➡️ Експортуємо всі відеоблоки окремо...")
        for i, block in enumerate(final_blocks):
            temp_path = os.path.join(temp_dir, f"block_{i}.mp4")
            print(f"  [{i + 1}/{len(final_blocks)}] -> {temp_path}")
            block.write_videofile(temp_path, codec="libx264", audio=False, fps=fps, threads=os.cpu_count(), remove_temp=True,
                                  logger="bar")
            temp_files.append(temp_path)
            block.close()
        concat_file = os.path.join(temp_dir, "all_clips.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for path in temp_files:
                abs_path = os.path.abspath(path).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")
        print("\n➡️ Склеюємо всі блоки через ffmpeg...")
        concat_out = output_path + ".noaudio.mp4"
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            concat_out
        ]
        subprocess.run(concat_cmd, check=True)
        print(f"✅ Відео без аудіо збережено: {concat_out}")
        if track_audio_path:
            print("➡️ Додаємо аудіо...")
            add_audio_cmd = [
                "ffmpeg", "-y",
                "-i", concat_out,
                "-i", track_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]
            subprocess.run(add_audio_cmd, check=True)
            print(f"✅ Фінальний кліп з аудіо збережено: {output_path}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not final_blocks:
        print("❌ Жоден блок не згенеровано.")
        sys.exit(1)
    print(f"🎥 Готуємось експортувати {len(final_blocks)} блоків...")

    export_and_concat_blocks(
        final_blocks,
        output_path=OUTPUT_PATH,
        track_audio_path=TRACK_PATH,
        fps=FPS
    )

    # ⬇️ Ось тут зберігаємо порядок блоків!
    with open("chunk_paths.txt", "w", encoding="utf-8") as f:
        for path in chunk_paths_all:
            f.write(path + "\n")
    print("✅ Збережено chunk_paths.txt")

    import gc

    gc.collect()

    if not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) < 5_000_000:
        print("❌ Помилка: фінальний відеофайл не згенеровано або він занадто малий!")
        sys.exit(1)
    print(f"✅ Готово! Фінальне відео збережено: {OUTPUT_PATH}")


from fcpxml_generator import trim_chunks_by_blocks, prepare_trimmed_chunks_folder, generate_fcpxml_trimmed, safe_close
from moviepy.editor import VideoFileClip

# 1. Дістаємо тривалості всіх чанків (безпечний close!):
chunk_durations = []
for path in chunk_paths_all:
    clip = VideoFileClip(path)
    chunk_durations.append(clip.duration)
    safe_close(clip)

# 2. Підрізаємо останній chunk кожного артиста, якщо "перекриває" перехід:
trimmed_chunks = trim_chunks_by_blocks(chunk_paths_all, chunk_durations, artist_end_times, fps=FPS)

# 3. Готуємо папку з файлами під унікальними іменами (з урахуванням повторів):
prepared_chunk_paths, trimmed_durations = prepare_trimmed_chunks_folder(
    trimmed_chunks,
    target_folder="output/chunks_for_edit"
)

# 4. Генеруємо FCPXML для DaVinci:
generate_fcpxml_trimmed(
    prepared_chunk_paths,
    trimmed_durations,
    output_path="output/remix_timeline.fcpxml",
    fps=FPS
)
print("✅ FCPXML для DaVinci готовий!")