import os
from ultralytics import YOLO

# ----------------------------
# Project root
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute path to dataset.yaml
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "dataset.yaml")

# Absolute path to pretrained YOLO model
MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

# ----------------------------
# Training settings
# ----------------------------
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 640
RUN_NAME = "vehicle_detector"

# ----------------------------
# Train
# ----------------------------
def main():
    print(f"Using dataset: {DATA_YAML}")
    print(f"Using model: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        name=RUN_NAME,
        project=os.path.join(PROJECT_ROOT, "runs", "train"),
        exist_ok=True,
        verbose=True
    )

if __name__ == "__main__":
    main()