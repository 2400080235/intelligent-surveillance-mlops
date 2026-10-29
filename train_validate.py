
import random
import shutil
import re
from pathlib import Path

import mlflow
from ultralytics import YOLO, settings


# ---------------- PROJECT PATHS ----------------

PROJECT_DIR = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_DIR / "data" / "raw" / "coco128"
DEMO_DIR = PROJECT_DIR / "data" / "training_demo"

IMAGE_SOURCE = SOURCE_DIR / "images" / "train2017"
LABEL_SOURCE = SOURCE_DIR / "labels" / "train2017"

TRAIN_DIR = DEMO_DIR / "images" / "train"
VAL_DIR = DEMO_DIR / "images" / "val"

TRAIN_LABELS = DEMO_DIR / "labels" / "train"
VAL_LABELS = DEMO_DIR / "labels" / "val"

DATA_YAML = DEMO_DIR / "coco128.yaml"

MODEL_PATH = PROJECT_DIR / "yolo11n.pt"

OUTPUT_DIR = (
    PROJECT_DIR / "runs" / "train" / "surveillance_demo"
)

BEST_MODEL_PATH = OUTPUT_DIR / "weights" / "best.pt"

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Intelligent Surveillance System"


# ---------------- CREATE DIRECTORIES ----------------

for folder in [
    TRAIN_DIR,
    VAL_DIR,
    TRAIN_LABELS,
    VAL_LABELS,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------- CHECK SOURCE DATA ----------------

if not IMAGE_SOURCE.exists():
    raise FileNotFoundError(
        f"Image folder not found: {IMAGE_SOURCE}"
    )

if not LABEL_SOURCE.exists():
    raise FileNotFoundError(
        f"Label folder not found: {LABEL_SOURCE}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Pretrained model not found: {MODEL_PATH}"
    )


# ---------------- FIND MATCHING IMAGE-LABEL PAIRS ----------------

pairs = []

for image_path in IMAGE_SOURCE.iterdir():
    if image_path.suffix.lower() not in [
        ".jpg",
        ".jpeg",
        ".png",
    ]:
        continue

    label_path = LABEL_SOURCE / (
        image_path.stem + ".txt"
    )

    if label_path.exists():
        pairs.append((image_path, label_path))

print("Matched image-label pairs:", len(pairs))

if len(pairs) < 2:
    raise RuntimeError(
        "Not enough matching images and labels were found."
    )


# ---------------- SPLIT DATASET ----------------

random.seed(42)
random.shuffle(pairs)

split_index = int(len(pairs) * 0.8)

train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

print("Training images:", len(train_pairs))
print("Validation images:", len(val_pairs))


# ---------------- COPY DATASET ----------------

def copy_pairs(pair_list, image_folder, label_folder):
    for image_path, label_path in pair_list:
        shutil.copy2(
            image_path,
            image_folder / image_path.name,
        )

        shutil.copy2(
            label_path,
            label_folder / label_path.name,
        )


copy_pairs(train_pairs, TRAIN_DIR, TRAIN_LABELS)
copy_pairs(val_pairs, VAL_DIR, VAL_LABELS)


# ---------------- CREATE YOLO YAML ----------------

yaml_text = f"""
path: {DEMO_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""

DATA_YAML.write_text(
    yaml_text.strip(),
    encoding="utf-8",
)


# ---------------- CONFIGURE MLFLOW ----------------

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Prevent conflicting automatic MLflow runs from Ultralytics.
settings.update({"mlflow": False})

print("MLflow tracking URI:", MLFLOW_URI)
print("Experiment:", EXPERIMENT_NAME)


# ---------------- TRAIN AND VALIDATE ----------------

with mlflow.start_run(
    run_name="YOLO_Training_Validation_Demo"
):

    mlflow.log_param("model", "YOLO11n")
    mlflow.log_param(
        "dataset",
        "COCO128 matched image-label pairs",
    )
    mlflow.log_param(
        "training_images",
        len(train_pairs),
    )
    mlflow.log_param(
        "validation_images",
        len(val_pairs),
    )
    mlflow.log_param("epochs", 1)
    mlflow.log_param("image_size", 320)
    mlflow.log_param("batch_size", 4)
    mlflow.log_param("device", "cpu")

    print("\nStarting YOLO training...")

    model = YOLO(str(MODEL_PATH))

    model.train(
        data=str(DATA_YAML),
        epochs=1,
        imgsz=320,
        batch=4,
        workers=0,
        device="cpu",
        project=str(PROJECT_DIR / "runs" / "train"),
        name="surveillance_demo",
        exist_ok=True,
        plots=False,
        verbose=True,
    )

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {BEST_MODEL_PATH}"
        )

    print("\nTraining completed!")
    print("Best model:", BEST_MODEL_PATH)

    print("\nStarting validation...")

    trained_model = YOLO(str(BEST_MODEL_PATH))

    results = trained_model.val(
        data=str(DATA_YAML),
        imgsz=320,
        batch=4,
        workers=0,
        device="cpu",
        plots=False,
        verbose=True,
    )

    # ---------------- LOG VALIDATION METRICS ----------------

    metrics = results.results_dict

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            safe_key = re.sub(
                r"[^a-zA-Z0-9_./ -]",
                "_",
                key,
            )

            mlflow.log_metric(
                safe_key,
                float(value),
            )

    # ---------------- LOG ARTIFACTS ----------------

    mlflow.log_artifact(str(DATA_YAML))
    mlflow.log_artifact(str(BEST_MODEL_PATH))

    print("\nTraining and validation completed!")
    print("MLflow experiment:", EXPERIMENT_NAME)
    print("Best model:", BEST_MODEL_PATH)
    print("Open MLflow:", MLFLOW_URI)
