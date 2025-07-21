import os
import cv2
import mediapipe as mp
import numpy as np
import json

def is_mouth_moving_clip(video_path, threshold=0.012):
    cap = cv2.VideoCapture(video_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)
    mouth_movements = []
    found_faces = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            found_faces += 1
            landmarks = results.multi_face_landmarks[0].landmark
            mouth_top = np.array([landmarks[13].x, landmarks[13].y])
            mouth_bottom = np.array([landmarks[14].x, landmarks[14].y])
            distance = np.linalg.norm(mouth_top - mouth_bottom)
            mouth_movements.append(distance)
    cap.release()
    face_mesh.close()
    print(f"---- {video_path} ----")
    print(f"Кадрів з обличчям: {found_faces}")
    print(f"mouth_movements: {mouth_movements}")
    if len(mouth_movements) < 2:
        print("Недостатньо руху губ")
        return 0, 0.0  # <--- повертаємо mean_moving = 0.0
    diffs = [abs(mouth_movements[i] - mouth_movements[i-1]) for i in range(1, len(mouth_movements))]
    #print(f"diffs: {diffs}")
    moving_frames = [d > threshold for d in diffs]
    mean_moving = np.mean(moving_frames)
    print(f"mean_moving: {mean_moving}")
    return int(mean_moving > 0.05), float(mean_moving)   # <--- тут

def classify_all_chunks(chunk_dir="temp/video_chunks"):
    chunk_info = []
    for artist in os.listdir(chunk_dir):
        artist_path = os.path.join(chunk_dir, artist)
        for fname in os.listdir(artist_path):
            if not fname.endswith(".mp4"):
                continue
            chunk_path = os.path.join(artist_path, fname)
            print(f"Класифікую: {chunk_path}")
            is_mouth_moving, mean_moving = is_mouth_moving_clip(chunk_path)
            chunk_info.append({"path": chunk_path, "is_mouth_moving": is_mouth_moving, "mean_moving": mean_moving})
    return chunk_info

if __name__ == "__main__":
    results = classify_all_chunks()
    os.makedirs("temp", exist_ok=True)
    with open("temp/chunk_lipsync_info.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Результати збережені у temp/chunk_lipsync_info.json")
