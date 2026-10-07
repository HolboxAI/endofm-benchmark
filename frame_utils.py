"""Shared frame-extraction helper. Direct-seek, not sequential decode -- fine for
grabbing a handful of frames to look at, not for dense per-second extraction."""
import cv2
from PIL import Image


def extract_frame(video_path, timestamp_s: float) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame at t={timestamp_s}s from {video_path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def extract_frames(video_path, timestamps_s):
    return [extract_frame(video_path, t) for t in timestamps_s]
