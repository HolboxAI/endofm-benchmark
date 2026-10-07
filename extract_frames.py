"""Standalone frame extraction (mirrors notebook section 3). Run once to populate
data/kvasir-capsule-labeled-videos/extracted_frames/ before executing the notebook,
so the notebook's own extraction cell just finds everything already cached."""
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(r"E:/endofm-benchmark")
DATA_DIR = PROJECT_ROOT / "data"
FRAMES_DIR = DATA_DIR / "kvasir-capsule-labeled-videos" / "extracted_frames"
SPLITS_DIR = DATA_DIR / "official_splits"
RAW_DIR = DATA_DIR / "kvasir-capsule-labeled-videos" / "raw"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
video_paths = [p for p in RAW_DIR.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
video_id_to_path = {p.stem: p for p in video_paths}
print(f"Found {len(video_id_to_path)} video files.")


def load_split(name):
    rows = []
    with open(SPLITS_DIR / f"{name}.csv", newline="") as f:
        for row in csv.DictReader(f):
            video_id, frame_no = row["filename"].rsplit(".", 1)[0].rsplit("_", 1)
            rows.append((video_id, int(frame_no), row["label"]))
    return rows


split0 = load_split("split_0")
split1 = load_split("split_1")
print(f"split_0: {len(split0)} labeled frames | split_1: {len(split1)} labeled frames")

wanted_by_video = defaultdict(set)
for video_id, frame_no, _ in split0 + split1:
    wanted_by_video[video_id].add(frame_no)

missing = [vid for vid in wanted_by_video if vid not in video_id_to_path]
if missing:
    print(f"WARNING: {len(missing)} referenced video ids not found: {missing[:5]}")


def extract_frames_for_video(video_id, frame_numbers):
    out_dir = FRAMES_DIR / video_id
    remaining = {fn for fn in frame_numbers if not (out_dir / f"{fn}.jpg").exists()}
    if not remaining:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_id_to_path[video_id]))
    idx = 0
    target_max = max(remaining)
    while idx <= target_max:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in remaining:
            cv2.imwrite(str(out_dir / f"{idx}.jpg"), frame)
        idx += 1
    cap.release()


for video_id, frame_numbers in tqdm(wanted_by_video.items(), desc="Extracting frames per video"):
    if video_id in video_id_to_path:
        extract_frames_for_video(video_id, frame_numbers)

n_extracted = sum(1 for _ in FRAMES_DIR.rglob("*.jpg"))
print(f"Total extracted frames on disk: {n_extracted}")
