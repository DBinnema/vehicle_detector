import os
import cv2
import random

# Map YOLO class IDs to names (must match dataset.yaml)
CLASS_NAMES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "bus"
}

# Path to YOLO dataset (adjust as needed)
IMAGE_DIR = "../data/visdrone_yolo/images/train"
LABEL_DIR = "../data/visdrone_yolo/labels/train"

# List all images
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

# Randomly shuffle images
random.shuffle(image_files)

def draw_yolo_boxes(image, label_path):
    h, w = image.shape[:2]
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x_c, y_c, bw, bh = map(float, parts)
            cls_id = int(cls_id)
            # Convert normalized YOLO coords to pixel coords
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put class name
            cv2.putText(image, CLASS_NAMES.get(cls_id, str(cls_id)),
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
    return image

# Preview N images
N = 10
for img_file in image_files[:N]:
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, img_file.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    img = draw_yolo_boxes(img, label_path)
    cv2.imshow("YOLO Label Preview", img)
    key = cv2.waitKey(0)
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()