import os
import json
import cv2
from src.frame_extractor import extract_frames
from src.vllm_reasoner import run_vllm_verification


def run_pipeline(video_path: str, out_dir: str, golden_steps: list,
                 every_n_frames: int = 8, use_api: bool = False, api_key: str = None):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "verification_result.json")

    # --- If JSON already exists, just load it ---
    if os.path.exists(out_path):
        print(f"📂 Using cached verification result from {out_path}")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --- Otherwise run the pipeline ---
    print("Extracting frames...")
    frames, timestamps = extract_frames(video_path, out_dir, every_n_frames)
    print(f"Extracted {len(frames)} frames to {out_dir}")

    print("Running VLLM verification...")
    result = run_vllm_verification(frames, golden_steps, use_api=use_api, api_key=api_key)

    verification = result["verification"]

    # Add timestamps + annotate frames
    for step, info in verification.items():
        frame_file = info.get("evidence_frame")
        if frame_file and frame_file in frames:
            try:
                frame_idx = frames.index(frame_file)
                timestamp = timestamps[frame_idx]
                info["timestamp"] = f"{timestamp} sec"

                img = cv2.imread(frame_file)
                if img is not None:
                    cv2.putText(img, f"Step {step}: {info['status']}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255) if info['status'] != "done" else (0, 255, 0), 2)
                    annotated_path = os.path.join(out_dir, f"annotated_step{step}.jpg")
                    cv2.imwrite(annotated_path, img)
                    info["annotated_frame"] = annotated_path
            except Exception as e:
                print(f"⚠️ Could not annotate frame for step {step}: {e}")

    # --- Compare with golden reference ---
    golden_path = "out_golden/verification_result.json"
    if os.path.exists(golden_path):
        print("🔗 Comparing with golden reference...")
        with open(golden_path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        compared = {}
        for step, g_info in golden["verification"].items():
            t_info = verification.get(step, {})
            compared[step] = {
                "expected": g_info["expected"],
                "status": t_info.get("status", "missing"),
                "note": t_info.get("note"),
                "timestamp": t_info.get("timestamp"),
                "annotated_frame": t_info.get("annotated_frame"),
                "golden_frame": g_info.get("evidence_frame")
            }
        verification = compared

    # --- Save result JSON ---
    out_json = {
        "video": video_path,
        "frames": frames,
        "verification": verification,
        "vllm_texts": result["answers"]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"✅ Saved verification result to {out_path}")
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
