from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


def process():
    print("Stacking mode selected.")
    print()
    print("1. Stack images from a folder")
    print("2. Extract frames from a video, then stack them")
    print("0. Back to main menu")

    choice = input("Choose a stacking option: ").strip()

    if choice == "1":
        folder_path = input("Enter the folder path: ").strip()
        stack_folder(folder_path)

    elif choice == "2":
        video_path = input("Enter the video path: ").strip()
        stack_video(video_path)

    elif choice == "0":
        print("Returning to main menu.")

    else:
        print("Invalid choice.")


def stack_folder(folder_path):
    folder_path = clean_path(folder_path)
    folder = Path(folder_path)

    if not folder.exists():
        print("Could not find that folder. Check the path and try again.")
        return

    if not folder.is_dir():
        print("That path is not a folder.")
        return

    image_paths = find_image_files(folder)
    output_folder = folder / "outputs"
    stack_and_save(image_paths, output_folder, f"{folder.name}_stacked.png")


def stack_video(video_path):
    video_path = clean_path(video_path)
    video = Path(video_path)

    if not video.exists():
        print("Could not find that video. Check the path and try again.")
        return

    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        print("That file does not look like a supported video file.")
        return

    frame_step = ask_for_int("Save every how many frames", 10)
    max_frames = ask_for_int("Maximum frames to save", 200)

    output_folder = video.parent / "outputs"
    frames_folder = output_folder / f"{video.stem}_frames"
    frames_folder.mkdir(parents=True, exist_ok=True)

    frame_paths = extract_video_frames(video, frames_folder, frame_step, max_frames)

    if len(frame_paths) == 0:
        print("No frames were extracted from the video.")
        return

    stack_and_save(frame_paths, output_folder, f"{video.stem}_video_stacked.png")


def extract_video_frames(video, frames_folder, frame_step, max_frames):
    capture = cv2.VideoCapture(str(video))

    if not capture.isOpened():
        print("Could not open the video file.")
        return []

    frame_paths = []
    frame_number = 0
    saved_count = 0

    while saved_count < max_frames:
        success, frame = capture.read()

        if not success:
            break

        if frame_number % frame_step == 0:
            frame_path = frames_folder / f"frame_{saved_count + 1:04}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            saved_count += 1

        frame_number += 1

    capture.release()

    print(f"Saved {len(frame_paths)} frame(s) to: {frames_folder}")
    return frame_paths


def stack_and_save(image_paths, output_folder, output_name):
    if len(image_paths) == 0:
        print("No image files found.")
        return

    if len(image_paths) == 1:
        print("Only one image found. Stacking works best with many images.")

    print(f"Found {len(image_paths)} image(s).")
    print("Stacking images...")

    stacked_image = stack_images(image_paths)

    if stacked_image is None:
        print("Could not stack the images.")
        return

    output_folder.mkdir(exist_ok=True)

    output_path = output_folder / output_name
    stacked_image.save(output_path)

    print()
    print(f"Stacked image saved: {output_path}")
    stacked_image.show(title="Stacked Image")


def clean_path(path):
    return path.strip().strip('"').strip("'")


def ask_for_int(setting_name, default_value):
    user_input = input(f"{setting_name} [{default_value}]: ").strip()

    if user_input == "":
        return default_value

    try:
        number = int(user_input)
    except ValueError:
        print(f"Invalid number. Using {default_value}.")
        return default_value

    if number < 1:
        print(f"Number must be at least 1. Using {default_value}.")
        return default_value

    return number


def find_image_files(folder):
    image_paths = []

    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    return sorted(image_paths)


def stack_images(image_paths):
    first_image = open_image(image_paths[0])

    if first_image is None:
        return None

    target_size = first_image.size
    image_arrays = []

    for path in image_paths:
        image = open_image(path)

        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue

        if image.size != target_size:
            print(f"Resizing {path.name} to match the first image.")
            image = image.resize(target_size)

        image_arrays.append(np.array(image, dtype=np.float32))

    if len(image_arrays) == 0:
        return None

    average_array = np.mean(image_arrays, axis=0)
    average_array = np.clip(average_array, 0, 255).astype(np.uint8)

    return Image.fromarray(average_array)


def open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None
