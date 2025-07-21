import os
import shutil
from moviepy.editor import VideoFileClip

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

def path_to_uri(path):
    abspath = os.path.abspath(path)
    return "file:///" + abspath.replace("\\", "/")

def frames_str(frames, fps):
    return f"{frames}/{fps}s"

def trim_chunks_by_blocks(chunk_paths, chunk_durations, block_end_times, fps=30):
    """
    chunk_paths: [chunk1, chunk2, ...] (по фінальному порядку)
    chunk_durations: [dur1, dur2, ...] (тривалість кожного chunk у секундах)
    block_end_times: [end1, end2, ...] (час закінчення кожного блоку, наприклад, [93, 233, ...] сек)
    Повертає: [(chunk_path, use_dur), ...] (use_dur — скільки брати з цього chunk)
    """
    trimmed = []
    i = 0
    prev_end = 0
    for block_end in block_end_times:
        block_len = block_end - prev_end
        curr_sum = 0
        while i < len(chunk_paths) and curr_sum < block_len:
            left = block_len - curr_sum
            use_dur = min(chunk_durations[i], left)
            trimmed.append((chunk_paths[i], use_dur))
            curr_sum += use_dur
            if use_dur < chunk_durations[i]:
                # цей chunk треба обрізати, наступний блок починається з нового артиста
                i += 1
                break
            i += 1
        prev_end = block_end
    # Якщо залишились чанки після останнього блоку — додаємо їх повністю
    while i < len(chunk_paths):
        trimmed.append((chunk_paths[i], chunk_durations[i]))
        i += 1
    return trimmed

def prepare_trimmed_chunks_folder(trimmed_chunks, target_folder="output/chunks_for_edit"):
    # Копіює та підрізає чанки (ffmpeg копіює швидко)
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)
    result_paths = []
    durations = []
    for i, (chunk_path, use_dur) in enumerate(trimmed_chunks):
        base = os.path.basename(chunk_path)
        dst = os.path.join(target_folder, f"{i:03d}__{base}")
        orig_dur = VideoFileClip(chunk_path).duration
        safe_close(VideoFileClip(chunk_path))
        # Якщо треба підрізати chunk:
        if use_dur < orig_dur - 0.05:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", chunk_path,
                "-t", str(use_dur),
                "-c", "copy",
                dst
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.copy2(chunk_path, dst)
        result_paths.append(dst)
        durations.append(use_dur)
    print(f"✅ {len(result_paths)} trimmed chunks copied to {target_folder}")
    return result_paths, durations

def generate_fcpxml_trimmed(chunk_paths, durations, output_path, fps=30):
    import xml.etree.ElementTree as ET

    fcpxml = ET.Element('fcpxml', version="1.6")
    resources = ET.SubElement(fcpxml, 'resources')
    ET.SubElement(resources, 'format',
        id="r1",
        name=f"FFVideoFormat1080p{fps}",
        frameDuration=f"1/{int(fps)}s",
        width="1920", height="1080"
    )

    unique_asset_counter = 2
    asset_ids = []
    frames_list = []
    for path, dur in zip(chunk_paths, durations):
        asset_id = f"r{unique_asset_counter}"
        unique_asset_counter += 1
        frames = int(round(dur * fps))
        ET.SubElement(resources, 'asset',
            id=asset_id,
            name=os.path.basename(path),
            start="0s",
            duration=f"{frames}/{fps}s",
            hasVideo="1",
            hasAudio="1",
            format="r1",
            src=path_to_uri(path)
        )
        asset_ids.append(asset_id)
        frames_list.append(frames)

    total_frames = sum(frames_list)
    library = ET.SubElement(fcpxml, 'library')
    event = ET.SubElement(library, 'event', name="Remix Event")
    project = ET.SubElement(event, 'project', name="Remix Project")
    sequence = ET.SubElement(project, 'sequence', duration=f"{total_frames}/{fps}s", format="r1")
    spine = ET.SubElement(sequence, 'spine')

    offset_frames = 0
    for idx, asset_id in enumerate(asset_ids):
        frames = frames_list[idx]
        ET.SubElement(spine, 'asset-clip',
            ref=asset_id,
            name=os.path.basename(chunk_paths[idx]),
            offset=f"{offset_frames}/{fps}s",
            duration=f"{frames}/{fps}s",
            start="0s",
            audioRole="dialogue"
        )
        offset_frames += frames

    tree = ET.ElementTree(fcpxml)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"✅ FCPXML згенеровано для DaVinci: {output_path}")

# ===============================
# Як використовувати у main:
# ===============================
# 1. chunk_paths_all = [...]  # список всіх чанків у фінальному порядку (з повторами, якщо треба)
# 2. artist_end_times = [...] # таймінги переходу артистів у секундах (напр. [93, 233, ...])
# 3. FPS = 30

# -- В main (після формування chunk_paths_all та artist_end_times) -- #

