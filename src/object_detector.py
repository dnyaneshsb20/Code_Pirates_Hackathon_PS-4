# src/object_detector.py
import cv2
import os
import numpy as np

BASE_PATH = "data/objects"
TEMPLATES = {
    "case": cv2.imread(os.path.join(BASE_PATH, "case.jpg"), 0),
    "left_earbud": cv2.imread(os.path.join(BASE_PATH, "left_earbud.jpg"), 0),
    "right_earbud": cv2.imread(os.path.join(BASE_PATH, "right_earbud.jpg"), 0),
    "cable": cv2.imread(os.path.join(BASE_PATH, "cable.jpg"), 0),
}

def detect_objects(image_path: str, threshold: float = 0.45):
    detected = []
    frame = cv2.imread(image_path, 0)
    if frame is None:
        return detected

    fh, fw = frame.shape[:2]

    for name, template in TEMPLATES.items():
        if template is None:
            print(f"⚠️ template missing: {name}")
            continue

        best_val = 0.0

        # Try multiple scales and rotations
        for scale in [0.5, 0.75, 1.0, 1.25]:
            h, w = template.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w < 8 or new_h < 8:
                continue
            if new_w > fw or new_h > fh:
                # scale down to fit frame
                scale2 = min(fw / new_w, fh / new_h) * 0.9
                new_w = max(8, int(new_w * scale2))
                new_h = max(8, int(new_h * scale2))

            resized = cv2.resize(template, (new_w, new_h))

            for angle in [0, 90, 180, 270]:
                M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
                rotated = cv2.warpAffine(resized, M, (new_w, new_h))
                if rotated.shape[0] > fh or rotated.shape[1] > fw:
                    continue

                res = cv2.matchTemplate(frame, rotated, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = float(max_val)

        # Debug print so you can see scores in console
        print(f"Template {name} best match: {best_val:.2f}")

        # Append as dict (consistent API)
        if best_val >= threshold:
            detected.append({"object": name, "confidence": best_val})

    return detected
