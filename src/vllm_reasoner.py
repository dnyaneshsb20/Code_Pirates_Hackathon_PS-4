import os
import json
import random
from typing import List, Dict

# --------------------------
# SIMULATED VLLM (for demo)
# --------------------------
def simulated_vllm(image_paths: List[str], prompt: str, mode: str = "basic") -> Dict[str, str]:
    responses = {}
    for p in image_paths:
        base = os.path.basename(p)
        r = random.random()
        if r < 0.6:
            if "close" in p.lower() or "after" in p.lower():
                txt = "Case closed and LED appears on. Charging likely started."
            else:
                txt = "Both earbuds observed in correct slots; orientation looks fine."
        elif r < 0.85:
            txt = "One earbud looks absent or misaligned in view."
        else:
            txt = "Uncertain due to blur/occlusion."
        responses[p] = txt
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
def verify_steps_with_vllm(image_answers: Dict[str, str], golden_steps: List[str]) -> Dict[str, Dict]:
    out = {}
    for i, step in enumerate(golden_steps, start=1):
        out[str(i)] = {"expected": step, "status": "uncertain", "evidence_frame": None, "note": None}

    for frame, text in image_answers.items():
        low = text.lower()

        # ---------------- Step-specific keyword rules ----------------
        if "earbud" in low and "cable" in low and "case" in low:
            out["1"]["status"] = "done"
            out["1"]["evidence_frame"] = frame
            out["1"]["note"] = text

        elif "open" in low and "case" in low:
            out["2"]["status"] = "done"
            out["2"]["evidence_frame"] = frame
            out["2"]["note"] = text

        elif "left" in low and "earbud" in low:
            out["3"]["status"] = "done"
            out["3"]["evidence_frame"] = frame
            out["3"]["note"] = text

        elif "right" in low and "earbud" in low:
            out["4"]["status"] = "done"
            out["4"]["evidence_frame"] = frame
            out["4"]["note"] = text

        elif "closed" in low and "case" in low:
            out["5"]["status"] = "done"
            out["5"]["evidence_frame"] = frame
            out["5"]["note"] = text

        elif "cable" in low and ("connected" in low or "insert" in low or "plug" in low):
            out["6"]["status"] = "done"
            out["6"]["evidence_frame"] = frame
            out["6"]["note"] = text

        # ---------------- Otherwise keep original fallback ----------------
        else:
            if "missing" in low or "absent" in low:
                for i, step in enumerate(golden_steps, start=1):
                    if any(tok in low for tok in step.lower().split()):
                        out[str(i)]["status"] = "missing"
                        out[str(i)]["evidence_frame"] = frame
                        out[str(i)]["note"] = text
            else:
                for i, step in enumerate(golden_steps, start=1):
                    if out[str(i)]["note"] is None:
                        out[str(i)]["note"] = text

    return out



from src.object_detector import detect_objects   # new file we’ll create

# --------------------------
# Convenience wrapper
# --------------------------
def run_vllm_verification(image_paths: List[str], golden_steps: List[str], use_api: bool = False, api_key: str = None) -> Dict:
    """
    For each image path:
      - run object detector (detect_objects returns list of dicts: {"object":..., "confidence":...})
      - compose a short object summary for the VLLM prompt
      - call either real VLLM or simulated VLLM
      - collect results and produce verification
    Returns a dict with keys: answers (mapping image->model-text-or-json), verification, detections (last frame)
    """
    results = {}
    all_detections_for_return = []  # keep last-frame detections to return (Streamlit expects this)

    for p in image_paths:
        # 1) object detection
        detected_objs = detect_objects(p)  # list of dicts {"object","confidence"}
        print("🔍 Detected objects:", detected_objs)
        all_detections_for_return = detected_objs

        # normalize names for logic
        detected_names = [d["object"] for d in detected_objs]

        # step1 logic: require case + cable + at least one earbud (practical demo rule)
        step1_ok = ("case" in detected_names and "cable" in detected_names and
                    ("left_earbud" in detected_names or "right_earbud" in detected_names))

        # build object summary string for VLLM prompt
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
        results[p] = answers[p]

        # 3) override step1 result based on detector if applicable
        if step1_ok:
            # override the frame's result to a JSON string the verifier can parse
            results[p] = json.dumps({
                "detected_step": 1,
                "status": "done",
                "note": "Case + cable + at least one earbud observed"
            })

    # 4) produce verification (parses the model outputs)
    verification = verify_steps_with_vllm(results, golden_steps)

    # Return answers mapping, verification structure, and last-detected-objects list
    return {"answers": results, "verification": verification, "detections": all_detections_for_return}
