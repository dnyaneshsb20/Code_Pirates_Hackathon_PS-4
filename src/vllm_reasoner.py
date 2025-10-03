# src/vllm_reasoner.py
import os
import json
import random
from typing import List, Dict

# --------------------------
# SIMULATED VLLM (for demo)
# --------------------------
def simulated_vllm(image_paths: List[str], prompt: str, mode: str = "basic") -> Dict[str, str]:
    """
    Very lightweight simulation that returns plausible textual responses for each frame.
    Mode 'basic' returns random yet sensible responses.
    This lets you demo the pipeline without API access.
    """
    responses = {}
    for p in image_paths:
        # Basic heuristic: use filename index to vary replies
        base = os.path.basename(p)
        # Randomize a bit so not all the same
        r = random.random()
        # Create fake messages mentioning keywords we expect (present/missing/uncertain)
        if r < 0.6:
            # mostly 'present' messages
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
# API template (fill in)
# --------------------------
def vllm_query_api_template(image_paths: List[str], prompt: str, api_key: str = None) -> Dict[str, str]:
    """
    TEMPLATE: Replace contents with real API call to your VLLM (OpenAI/GPT-4o or HuggingFace).
    Return mapping: image_path -> textual response from model.
    """
    # Example pseudocode (non-functional). You must replace with actual endpoint/client code.
    answers = {}
    for p in image_paths:
        answers[p] = "API_RESPONSE_PLACEHOLDER: replace this with actual API output"
    return answers

# --------------------------
# Verification function
# --------------------------
def verify_steps_with_vllm(image_answers: Dict[str, str], golden_steps: List[str]) -> Dict[str, Dict]:
    """
    Convert frame-level textual observations into step verification.
    Output format:
    {
      "1": {"expected": "Step text", "status": "done/missing/uncertain", "evidence_frame": "path", "note": "vllm_text"},
      ...
    }
    Heuristic-based, meant for demo. Replace or improve for production.
    """
    import re
    out = {}
    for i, step in enumerate(golden_steps, start=1):
        out[str(i)] = {"expected": step, "status": "uncertain", "evidence_frame": None, "note": None}

    # naive keyword matching — looks for words from the step in the VLLM text
    for frame, text in image_answers.items():
        low = text.lower()
        for i, step in enumerate(golden_steps, start=1):
            # pick first 3 meaningful tokens from step for match
            tokens = [t for t in re.findall(r"\w+", step.lower()) if len(t) > 2][:3]
            if not tokens:
                continue
            # if all tokens in text -> mark done
            if all(tok in low for tok in tokens):
                # promote to done if not already done
                if out[str(i)]["status"] != "done":
                    out[str(i)]["status"] = "done"
                    out[str(i)]["evidence_frame"] = frame
                    out[str(i)]["note"] = text
            else:
                # if text mentions missing or absent and includes one token -> missing
                if ("missing" in low or "absent" in low or "no" in low or "not" in low) and any(tok in low for tok in tokens):
                    out[str(i)]["status"] = "missing"
                    out[str(i)]["evidence_frame"] = frame
                    out[str(i)]["note"] = text
                else:
                    # if still nothing, leave uncertain but attach note if empty
                    if out[str(i)]["note"] is None:
                        out[str(i)]["note"] = text
    return out

# --------------------------
# Convenience wrapper
# --------------------------
def run_vllm_verification(image_paths: List[str], golden_steps: List[str], use_api: bool = False, api_key: str = None) -> Dict:
    prompt = f"Golden steps: " + " | ".join(golden_steps) + "\nFor each image, say which step is present or if missing."
    if use_api:
        answers = vllm_query_api_template(image_paths, prompt, api_key=api_key)
    else:
        answers = simulated_vllm(image_paths, prompt)
    verification = verify_steps_with_vllm(answers, golden_steps)
    return {"answers": answers, "verification": verification}

if __name__ == "__main__":
    # quick test
    frames = [f"data/frames_test/frame_{i:04d}.jpg" for i in range(6)]
    golden = [
        "Place left and right earbuds into slots",
        "Close the charging case completely",
        "Connect charging cable and check LED on",
        "Final packaging check, case sealed"
    ]
    out = run_vllm_verification(frames, golden, use_api=False)
    print(json.dumps(out["verification"], indent=2))
