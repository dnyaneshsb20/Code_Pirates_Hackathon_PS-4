# src/object_detector.py
import cv2
from ultralytics import YOLO

# Load YOLO model (use pretrained or your custom-trained weights)
# If you have trained on your 4 objects, replace "yolov8n.pt" with your weights path.
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

def detect_objects(image_input, threshold: float = 0.4):
    """
    Detect objects in a frame or image path using YOLOv8.
    Returns: list of dicts [{"object": name, "confidence": score}, ...]
    """

    # If input is a file path → read it
    if isinstance(image_input, str):
        frame = cv2.imread(image_input)
    else:
        # assume it's already a numpy frame (from webcam)
        frame = image_input

    if frame is None:
        return []

    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id]

            if conf >= threshold:
                detections.append({
                    "object": name,
                    "confidence": conf
                })

    return detections


# Quick test
if __name__ == "__main__":
    test_img = "test.jpg"  # replace with your image path
    dets = detect_objects(test_img)
    print("Detections:", dets)
