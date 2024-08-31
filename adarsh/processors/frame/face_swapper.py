import os
import urllib.request
import traceback
from typing import Any, List, Callable
import cv2
import threading
from insightface.model_zoo import get_model

import adarsh.globals
import adarsh.processors.frame.core
from adarsh.core import update_status
from adarsh.face_analyser import get_many_faces
from adarsh.typing import Frame, Face
from adarsh.utilities import is_image, is_video

FACE_SWAPPER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ADARSH.FACE-SWAPPER'
MODEL_URL = 'https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx'
MODEL_PATH = '/content/swap/models/inswapper_128.onnx'

def download_model(url: str, dest_path: str) -> None:
    try:
        if not os.path.isfile(dest_path):
            print(f"Downloading model from {url} to {dest_path}")
            urllib.request.urlretrieve(url, dest_path)
            print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        traceback.print_exc()

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            try:
                # Ensure the models directory exists
                os.makedirs('/content/swap/models', exist_ok=True)
                download_model(MODEL_URL, MODEL_PATH)
                # Load the model
                FACE_SWAPPER = get_model(MODEL_PATH, providers=['CPUExecutionProvider'])
            except Exception as e:
                print(f"Error initializing face swapper: {e}")
                traceback.print_exc()
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

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
        clear_face_swapper()
    except Exception as e:
        print(f"Error during post-process: {e}")
        traceback.print_exc()

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    try:
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    except Exception as e:
        print(f"Error swapping face: {e}")
        traceback.print_exc()
        return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    try:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
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
        adarsh.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"Error processing video: {e}")
        traceback.print_exc()
