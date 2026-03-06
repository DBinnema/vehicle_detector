# Vehicle Detector Project

This project trains a YOLO-based object detection model to detect vehicle types (car, van, truck, bus) using the VisDrone dataset and runs detection on live video.

---

# Project Setup

Follow these steps to set up the development environment and dataset.

---

# 1. Clone the Repository

```bash
git clone <repo-url>
cd vehicle_detector
```

---

# 2. Create a Python Virtual Environment

Create a project-specific environment:

```bash
python -m venv venv
```

Activate it.

### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```cmd
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

After activation your terminal should show:

```
(venv)
```

---

# 3. Install Project Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

Main dependencies include:

* Ultralytics YOLO
* OpenCV
* NumPy
* tqdm
* matplotlib

---

# 4. Download the VisDrone Dataset

Download the detection dataset from Kaggle:

https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset

Extract the dataset into:

```
data/visdrone_raw/
```

Expected structure:

```
data/
    visdrone_raw/
        VisDrone2019-DET-train/
            images/
            annotations/

        VisDrone2019-DET-val/
            images/
            annotations/
```

Do not modify files inside `visdrone_raw`. This folder contains the original dataset.

---

# 5. Convert Dataset to YOLO Format

Run the conversion script:

```bash
python scripts/convert_visdrone.py
```

This script will:

* read VisDrone annotations
* filter vehicle classes
* convert bounding boxes to YOLO format
* create a new dataset

Output dataset:

```
data/visdrone_yolo/

images/
    train/
    val/

labels/
    train/
    val/
```

---

# 6. Verify Dataset Configuration

The file `data/dataset.yaml` defines the dataset used for training.

Example:

```
path: data/visdrone_yolo
train: images/train
val: images/val

names:
  0: car
  1: van
  2: truck
  3: bus
```

---

# 7. Train the Model

Run the training script:

```bash
python scripts/train.py
```

Training results will be stored in:

```
runs/detect/train/
```

The best model weights will be saved as:

```
runs/detect/train/weights/best.pt
```

---

# 8. Run Live Detection

To test the trained model with a webcam:

```bash
python scripts/detect_live.py
```

This will open a window displaying detected vehicles in real time.

---

# Project Structure

```
vehicle_detector/

venv/                 Python environment
data/                 datasets
models/               trained models
scripts/              training and utility scripts
experiments/          experimental tests

requirements.txt      project dependencies
README.md             project documentation
```

---

# Notes

* Do not commit large datasets or trained models to Git.
* The `venv/` directory should remain local to each developer.
* The raw dataset should always remain unchanged so the processed dataset can be regenerated.
