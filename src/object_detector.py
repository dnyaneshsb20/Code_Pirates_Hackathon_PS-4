# src/object_detector.py
from ultralytics import YOLO
import cv2

MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

# Expand mapping (multiple COCO labels → your objects)
CLASS_MAP = {
    "cell phone": "case",
    "mouse": "case",
    "remote": "left_earbud",
    "earphone": "right_earbud",
    "tv": "cable",
    "keyboard": "cable",
    "laptop": "case",  # add extra fallbacks
}

def detect_objects(image_input, threshold: float = 0.25):  # lowered threshold
    if isinstance(image_input, str):
        frame = cv2.imread(image_input)
    else:
        frame = image_input

    if frame is None:
        return []

    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            coco_name = model.names[cls_id]

            # Debug print
            print(f"YOLO saw: {coco_name} ({conf:.2f})")

            # remap to your labels
            name = CLASS_MAP.get(coco_name, None)
            if name and conf >= threshold:
                detections.append({
                    "object": name,
                    "confidence": conf
                })

    return detections
