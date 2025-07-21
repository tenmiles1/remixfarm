# src/utils/temp_tools.py

import os
import shutil

def clear_temp_chunks(temp_dir="temp/video_chunks"):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

def temp_chunks_exist(temp_dir="temp/video_chunks"):
    # Повертає True, якщо в папці є підпапки/файли
    return any(os.scandir(temp_dir))
