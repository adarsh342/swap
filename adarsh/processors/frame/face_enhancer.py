from typing import Any, List, Callable
import cv2
import threading
import os
import urllib.request
import traceback

from gfpgan.utils import GFPGANer

import adarsh.globals
import adarsh.processors.frame.core
from adarsh.core import update_status
from adarsh.face_analyser import get_many_faces
from adarsh.typing import Frame, Face
from adarsh.utilities import is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ADARSH.FACE-ENHANCER'
MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'
MODEL_PATH = '/content/swap/models/GFPGANv1.4.pth'

def download_model(url: str, dest_path: str) -> None:
    try:
        if not os.path.isfile(dest_path):
            print(f"Downloading model from {url} to {dest_path}")
            urllib.request.urlretrieve(url, dest_path)
            print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        traceback.print_exc()

def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            try:
                # Ensure the models directory exists
                os.makedirs('/content/swap/models', exist_ok=True)
                download_model(MODEL_URL, MODEL_PATH)
                FACE_ENHANCER = GFPGANer(model_path=MODEL_PATH, upscale=1, device='cpu')
            except Exception as e:
                print(f"Error initializing face enhancer: {e}")
                traceback.print_exc()
    return FACE_ENHANCER

def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None

def pre_check() -> bool:
    try:
        if not os.path.isfile(MODEL_PATH):
            download_model(MODEL_URL, MODEL_PATH)
        return True
    except Exception as e:
        print(f"Error during pre-check: {e}")
        traceback.print_exc()
        return False

def pre_start() -> bool:
    try:
        if not is_image(adarsh.globals.target_path) and not is_video(adarsh.globals.target_path):
            update_status('Select an image or video for target path.', NAME)
            return False
        return True
    except Exception as e:
        print(f"Error during pre-start: {e}")
        traceback.print_exc()
        return False

def post_process() -> None:
    try:
        clear_face_enhancer()
    except Exception as e:
        print(f"Error during post-process: {e}")
        traceback.print_exc()

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    try:
        start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
        padding_x = int((end_x - start_x) * 0.5)
        padding_y = int((end_y - start_y) * 0.5)
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = max(0, end_x + padding_x)
        end_y = max(0, end_y + padding_y)
        temp_face = temp_frame[start_y:end_y, start_x:end_x]
        if temp_face.size:
            with THREAD_SEMAPHORE:
                _, _, temp_face = get_face_enhancer().enhance(
                    temp_face,
                    paste_back=True
                )
            temp_frame[start_y:end_y, start_x:end_x] = temp_face
        return temp_frame
    except Exception as e:
        print(f"Error enhancing face: {e}")
        traceback.print_exc()
        return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    try:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = enhance_face(target_face, temp_frame)
        return temp_frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        traceback.print_exc()
        return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    try:
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            result = process_frame(None, None, temp_frame)
            cv2.imwrite(temp_frame_path, result)
            if update:
                update()
    except Exception as e:
        print(f"Error processing frames: {e}")
        traceback.print_exc()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    try:
        target_frame = cv2.imread(target_path)
        result = process_frame(None, None, target_frame)
        cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    try:
        adarsh.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"Error processing video: {e}")
        traceback.print_exc()
