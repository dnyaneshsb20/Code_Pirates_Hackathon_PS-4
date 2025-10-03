# src/frame_extractor.py
import cv2
import os
from typing import List

def extract_frames(video_path: str, out_dir: str, every_n_frames: int = 10) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    saved = []
    idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            fname = os.path.join(out_dir, f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(fname, frame)
            saved.append(fname)
            saved_idx += 1
        idx += 1
    cap.release()
    return saved

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="frames_out")
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    frames = extract_frames(args.video, args.out, args.stride)
    print(f"Saved {len(frames)} frames to {args.out}")
