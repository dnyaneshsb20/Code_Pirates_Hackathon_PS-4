# src/vllm_reasoner.py
import os
import json
import random
from typing import List, Dict

# --------------------------
# SIMULATED VLLM (for demo)
def simulated_vllm(image_inputs, prompt, mode="basic"):
    responses = {}
    for i, p in enumerate(image_inputs):
        key = os.path.basename(p) if isinstance(p, str) else f"frame_{i}"

        r = random.random()
        if r < 0.2:
            txt = "Case, cable and both earbuds present on workstation."
        elif r < 0.35:
            txt = "Case opened fully."
        elif r < 0.5:
            txt = "Left earbud inserted in left slot."
        elif r < 0.65:
            txt = "Right earbud inserted in right slot."
        elif r < 0.8:
            txt = "Case closed completely."
        elif r < 0.95:
            txt = "Cable connected and LED on."
        else:
            txt = "Uncertain due to blur/occlusion."

        responses[key] = txt
    return responses



# --------------------------
# Lazy VLLM loader (loads model only when use_api=True)
# --------------------------
_model_loaded = False
_processor = None
_model = None
_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def ensure_model_loaded():
    global _model_loaded, _processor, _model
    if _model_loaded:
        return
    print("🔄 Loading VLLM (lazy):", _MODEL_ID)
    from transformers import LlavaForConditionalGeneration, LlavaProcessor
    import torch
    _processor = LlavaProcessor.from_pretrained(_MODEL_ID)
    _model = LlavaForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        device_map="auto",
        torch_dtype="auto"
    )
    _model_loaded = True
    print("✅ VLLM Loaded (lazy)")

# --------------------------
# REAL VLLM (HuggingFace LLaVA)
# #  
# def vllm_query_api_template(image_paths: List[str], prompt: str, api_key: str = None) -> Dict[str, str]:
#     """
#     Real implementation using HuggingFace LLaVA model.
#     For each image, returns a text response about observed step.
#     """
#     answers = {}
#     for p in image_paths:
#         image = Image.open(p).convert("RGB")
#         inputs = processor(
#         text=prompt,
#         images=image,
#         return_tensors="pt"
#         ).to(model.device)

#         output_ids = model.generate(**inputs, max_new_tokens=128)
#         text_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

#         answers[p] = text_output
#     return answers
def vllm_query_api_template(image_paths: List[str], prompt: str, api_key: str = None) -> Dict[str, str]:
    ensure_model_loaded()   # make sure model is loaded only if use_api=True
    answers = {}
    from PIL import Image
    for p in image_paths:
        image = Image.open(p).convert("RGB")
        inputs = _processor(text=prompt, images=image, return_tensors="pt").to(_model.device)
        output_ids = _model.generate(**inputs, max_new_tokens=128)
        text_output = _processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        answers[p] = text_output
    return answers


# --------------------------
# Verification function
# --------------------------
# def verify_steps_with_vllm(image_answers: Dict[str, str], golden_steps: List[str]) -> Dict[str, Dict]:
#     out = {}
#     for i, step in enumerate(golden_steps, start=1):
#         out[str(i)] = {"expected": step, "status": "uncertain", "evidence_frame": None, "note": None}

#     for frame, text in image_answers.items():
#         low = text.lower()

#         # ---------------- Step-specific keyword rules ----------------
#         if "earbud" in low and "cable" in low and "case" in low:
#             out["1"]["status"] = "done"
#             out["1"]["evidence_frame"] = frame
#             out["1"]["note"] = text

#         elif "open" in low and "case" in low:
#             out["2"]["status"] = "done"
#             out["2"]["evidence_frame"] = frame
#             out["2"]["note"] = text

#         elif "left" in low and "earbud" in low:
#             out["3"]["status"] = "done"
#             out["3"]["evidence_frame"] = frame
#             out["3"]["note"] = text

#         elif "right" in low and "earbud" in low:
#             out["4"]["status"] = "done"
#             out["4"]["evidence_frame"] = frame
#             out["4"]["note"] = text

#         elif "closed" in low and "case" in low:
#             out["5"]["status"] = "done"
#             out["5"]["evidence_frame"] = frame
#             out["5"]["note"] = text

#         elif "cable" in low and ("connected" in low or "insert" in low or "plug" in low):
#             out["6"]["status"] = "done"
#             out["6"]["evidence_frame"] = frame
#             out["6"]["note"] = text

#         # ---------------- Otherwise keep original fallback ----------------
#         else:
#             if "missing" in low or "absent" in low:
#                 for i, step in enumerate(golden_steps, start=1):
#                     if any(tok in low for tok in step.lower().split()):
#                         out[str(i)]["status"] = "missing"
#                         out[str(i)]["evidence_frame"] = frame
#                         out[str(i)]["note"] = text
#             else:
#                 for i, step in enumerate(golden_steps, start=1):
#                     if out[str(i)]["note"] is None:
#                         out[str(i)]["note"] = text

#     return out

# def verify_steps_with_vllm(image_answers, golden_steps, detections=None):
#     out = {str(i): {"expected": step, "status": "missing",
#                     "evidence_frame": None, "note": None}
#            for i, step in enumerate(golden_steps, start=1)}

#     seen = set()  # track accumulated objects across frames

#     for frame, text in image_answers.items():
#         det_objs = [] if detections is None else detections.get(frame, [])
#         objs = [d["object"] for d in det_objs]

#         # Step 1: Preparation (all objects must appear at least once across frames)
#         seen.update(objs)
#         if out["1"]["status"] == "missing" and all(x in seen for x in ["case", "cable", "left_earbud", "right_earbud"]):
#             out["1"].update({
#                 "status": "done",
#                 "evidence_frame": frame,
#                 "note": "Case, cable and both earbuds present on workstation."
#             })

#         # Step 2: Case open (if you keep text check or add explicit open detection later)
#         if out["2"]["status"] == "missing" and "open" in text.lower() and "case" in text.lower():
#             out["2"].update({
#                 "status": "done",
#                 "evidence_frame": frame,
#                 "note": "Case opened fully."
#             })

#         # Step 3: Left earbud inserted
#         if out["3"]["status"] == "missing" and "left_earbud" in objs:
#             out["3"].update({
#                 "status": "done",
#                 "evidence_frame": frame,
#                 "note": "Left earbud inserted in left slot."
#             })

#         # Step 4: Right earbud inserted
#         if out["4"]["status"] == "missing" and "right_earbud" in objs:
#             out["4"].update({
#                 "status": "done",
#                 "evidence_frame": frame,
#                 "note": "Right earbud inserted in right slot."
#             })

#         # Step 5: Case closed (still needs a text/detector condition)
#         if out["5"]["status"] == "missing" and "closed" in text.lower() and "case" in text.lower():
#             out["5"].update({
#                 "status": "done",
#                 "evidence_frame": frame,
#                 "note": "Case closed completely."
#             })

#         # Step 6: Cable connected
#         if out["6"]["status"] == "missing" and "cable" in objs:
#             out["6"].update({
#                 "status": "done",
#                 "evidence_frame": frame,
#                 "note": "Cable connected and LED on."
#             })

#     return out
def verify_steps_with_vllm(image_answers, golden_steps, detections=None):
    out = {
        str(i): {
            "expected": step,
            "status": "missing",
            "evidence_frame": None,
            "note": None
        }
        for i, step in enumerate(golden_steps, start=1)
    }

    # Track objects across frames for Step 1
    seen_objects = set()

    for frame, text in image_answers.items():
        det_objs = [] if detections is None else detections.get(frame, [])
        objs = [d["object"] for d in det_objs]

        # Update global object tracker
        seen_objects.update(objs)

        # --- Step 2: Open case ---
        if "case" in objs and "open" in text.lower():
            if out["2"]["status"] == "missing":
                out["2"].update({"status": "done", "evidence_frame": frame, "note": text})

        # --- Step 3: Left earbud ---
        if "left_earbud" in objs:
            if out["3"]["status"] == "missing":
                out["3"].update({"status": "done", "evidence_frame": frame, "note": "Left earbud inserted"})

        # --- Step 4: Right earbud ---
        if "right_earbud" in objs:
            if out["4"]["status"] == "missing":
                out["4"].update({"status": "done", "evidence_frame": frame, "note": "Right earbud inserted"})

        # --- Step 5: Case closed ---
        if "closed" in text.lower() and "case" in text.lower():
            if out["5"]["status"] == "missing":
                out["5"].update({"status": "done", "evidence_frame": frame, "note": text})

        # --- Step 6: Cable ---
        if "cable" in objs:
            if out["6"]["status"] == "missing":
                out["6"].update({"status": "done", "evidence_frame": frame, "note": "Cable plugged in"})

    # --- Step 1 (Preparation) – check across all frames ---
    if {"case", "left_earbud", "right_earbud", "cable"}.issubset(seen_objects):
        out["1"].update({"status": "done", "note": "Case, cable and both earbuds present"})

    return out


from src.object_detector import detect_objects   # new file we’ll create

# --------------------------
# Convenience wrapper
# --------------------------

def run_vllm_verification(frames, golden_steps, use_api: bool = False, api_key: str = None, raw: bool = False):
    """
    frames: list of either file paths OR raw frames (numpy arrays)
    """
    results = {}
    all_detections_for_return = []

    for i, p in enumerate(frames):
        detected_objs = detect_objects(p)
        print("🔍 Detected objects:", detected_objs)
        all_detections_for_return = detected_objs

        key = f"frame_{i}" if not isinstance(p, str) else os.path.basename(p)

        # Build object summary string
        if detected_objs:
            object_summary = ", ".join(f"{d['object']} (conf {d['confidence']:.2f})" for d in detected_objs)
        else:
            object_summary = "none"

        prompt = f"""
You are an AI assembly verification assistant.

Golden steps:
{chr(10).join(golden_steps)}

Evidence: {object_summary}

For this frame:
1. Identify which step (if any) is being performed.
2. Verify correctness (done, missing, out_of_order, uncertain).
3. Respond in JSON only:

{{
  "detected_step": <step number or null>,
  "status": "done" | "missing" | "out_of_order" | "uncertain",
  "note": "<short human-readable note>"
}}
"""

        # 2) call VLLM (real or simulated)
        try:
            if use_api:
                answers = vllm_query_api_template([p], prompt, api_key=api_key)
            else:
                answers = simulated_vllm([p], prompt)
        except Exception as e:
            print("⚠️ Falling back to simulated VLLM due to error:", e)
            answers = simulated_vllm([p], prompt)

        # store the raw answer (simulated returns text; real VLLM may return JSON-like text)
        results[key] = list(answers.values())[0] 

        # 3) override step1 result based on detector if applicable
        # if step1_ok:
        #     # override the frame's result to a JSON string the verifier can parse
        #     results[p] = json.dumps({
        #         "detected_step": 1,
        #         "status": "done",
        #         "note": "Case + cable + at least one earbud observed"
        #     })

    # 4) produce verification (parses the model outputs)
    verification = verify_steps_with_vllm(results, golden_steps)

    return {"answers": results, "verification": verification, "detections": all_detections_for_return}

