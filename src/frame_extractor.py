import cv2
import os
from typing import List, Tuple

def extract_frames(video_path: str, out_dir: str, every_n_frames: int = 10) -> Tuple[List[str], List[float]]:
    """
    Extract frames from a video and return both the frame file paths and their timestamps (in seconds).
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    timestamps = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % every_n_frames == 0:
            fname = os.path.join(out_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(fname, frame)
            frames.append(fname)

            # actual timestamp in seconds
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamps.append(round(ts, 2))

        idx += 1

    cap.release()
    return frames, timestamps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="frames_out")
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    frames, timestamps = extract_frames(args.video, args.out, args.stride)
    print(f"Saved {len(frames)} frames to {args.out}")
    print("Timestamps:", timestamps[:10], "...")
