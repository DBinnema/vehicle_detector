import os
import shutil
import cv2
from tqdm import tqdm

"""
convert_visdrone.py
├── constants (vehicle classes)
├── functions
│   ├── convert_bbox()
│   ├── process_annotation_file()
│   ├── process_split()  # train/val
├── main()
│   ├── define input/output paths
│   └── call process_split() for train and val
"""

# Vehicle classes mapping: VisDrone class ID -> YOLO class ID
VISDRONE_VEHICLE_CLASSES = {
    4: 0,  # car
    5: 1,  # van
    6: 2,  # truck
    9: 3   # bus
}

def convert_bbox(img_w, img_h, x, y, w, h):
    """
    Convert VisDrone bbox to YOLO normalized format.
    """
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def process_annotation_file(annotation_path, img_w, img_h):
    """
    Convert a single annotation file to YOLO format.
    Returns a list of YOLO lines.
    """
    yolo_lines = []

    with open(annotation_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue  # skip malformed lines

            x, y, w, h = map(float, parts[0:4])
            category = int(parts[5])

            if category not in VISDRONE_VEHICLE_CLASSES:
                continue

            yolo_class = VISDRONE_VEHICLE_CLASSES[category]
            x_c, y_c, w_n, h_n = convert_bbox(img_w, img_h, x, y, w, h)

            yolo_lines.append(f"{yolo_class} {x_c} {y_c} {w_n} {h_n}")

    return yolo_lines

def process_split(images_dir, annotations_dir, output_images, output_labels):
    """
    Convert all annotations in a dataset split (train/val)
    """
    
   
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]

    for annotation_file in tqdm(annotation_files, desc=f"Processing {os.path.basename(images_dir)}"):
        annotation_path = os.path.join(annotations_dir, annotation_file)
        image_name = annotation_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            continue  # skip if image missing

        img = cv2.imread(image_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        yolo_lines = process_annotation_file(annotation_path, img_w, img_h)

        if len(yolo_lines) == 0:
            continue  # skip images with no vehicles

        # Copy image
        shutil.copy(image_path, os.path.join(output_images, image_name))

        # Write YOLO label
        label_path = os.path.join(output_labels, annotation_file)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

def main():  
    
    
    # Project root (one level above scripts/)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Base input folder
    base_dir = os.path.join(PROJECT_ROOT, "data", "visdrone_raw")

    # Train paths (handle nested folder)
    train_dir = os.path.join(base_dir, "VisDrone2019-DET-train")
    train_images = os.path.join(train_dir, os.listdir(train_dir)[0], "images")         # assumes first nested folder
    train_ann = os.path.join(train_dir, os.listdir(train_dir)[0], "annotations")

    # Validation paths
    val_dir = os.path.join(base_dir, "VisDrone2019-DET-val")
    val_images = os.path.join(val_dir, os.listdir(val_dir)[0], "images")               # assumes first nested folder
    val_ann = os.path.join(val_dir, os.listdir(val_dir)[0], "annotations")

    # Output YOLO dataset
    output_base = os.path.join(PROJECT_ROOT, "data", "visdrone_yolo")

    # Process train
    process_split(
        train_images,
        train_ann,
        os.path.join(output_base, "images/train"),
        os.path.join(output_base, "labels/train")
    )

    # Process val
    process_split(
        val_images,
        val_ann,
        os.path.join(output_base, "images/val"),
        os.path.join(output_base, "labels/val")
    )

    print("Dataset conversion to YOLO format complete.")

if __name__ == "__main__":
    main()