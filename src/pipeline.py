# src/pipeline.py
import os
import json
from src.frame_extractor import extract_frames
from src.vllm_reasoner import run_vllm_verification

def run_pipeline(video_path: str, out_dir: str, golden_steps: list, every_n_frames: int = 8, use_api: bool = False, api_key: str = None):
    print("Extracting frames...")
    frames = extract_frames(video_path, out_dir, every_n_frames)
    print(f"Extracted {len(frames)} frames to {out_dir}")

    print("Running VLLM verification (simulated by default)...")
    result = run_vllm_verification(frames, golden_steps, use_api=use_api, api_key=api_key)

    out_json = {
        "video": video_path,
        "frames": frames,
        "verification": result["verification"],
        "vllm_texts": result["answers"]
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "verification_result.json")
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved verification result to {out_path}")
    return out_json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--outdir", default="out_frames")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--use_api", action="store_true")
    args = parser.parse_args()

    golden_steps = [
    "Step 1: Preparation – ensure case, left earbud, right earbud, and cable are present on workstation",
    "Step 2: Open the charging case fully, verify slots empty",
    "Step 3: Insert left earbud into left slot, align correctly",
    "Step 4: Insert right earbud into right slot, align correctly",
    "Step 5: Close the charging case fully, no gaps",
    "Step 6: Plug in charging cable, verify LED indicator ON"
    ]

    run_pipeline(args.video, args.outdir, golden_steps, args.stride, use_api=args.use_api, api_key=None)
