
from pathlib import Path
import csv
import mlflow

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = PROJECT_DIR / "runs" / "train" / "surveillance_demo" / "results.csv"
MODEL_FILE = PROJECT_DIR / "runs" / "train" / "surveillance_demo" / "weights" / "best.pt"
DATA_FILE = PROJECT_DIR / "data" / "training_demo" / "coco128.yaml"

# Check that the existing results and model are available
for file_path in [RESULTS_FILE, MODEL_FILE, DATA_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

# Read the existing final training row
with RESULTS_FILE.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise RuntimeError("No training results were found in results.csv")

last_row = rows[-1]

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Intelligent Surveillance System")

with mlflow.start_run(run_name="YOLO_Existing_Training_Results"):
    # Record parameters
    mlflow.log_param("model", "YOLO11n")
    mlflow.log_param("dataset", "COCO128")
    mlflow.log_param("epochs_recorded", len(rows))
    mlflow.log_param("image_size", 320)
    mlflow.log_param("training_images", 100)
    mlflow.log_param("validation_images", 26)
    mlflow.log_param("source", "Existing YOLO results.csv")

    # Record metrics from the existing YOLO results
    metric_mapping = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
    }

    for source_key, metric_name in metric_mapping.items():
        value = last_row.get(source_key)
        if value:
            mlflow.log_metric(metric_name, float(value))

    # Attach existing files as artifacts
    mlflow.log_artifact(str(RESULTS_FILE), artifact_path="training_results")
    mlflow.log_artifact(str(MODEL_FILE), artifact_path="model")
    mlflow.log_artifact(str(DATA_FILE), artifact_path="dataset")

    print("Existing training results logged successfully!")
    print("Run ID:", mlflow.active_run().info.run_id)
    print("Precision:", last_row["metrics/precision(B)"])
    print("Recall:", last_row["metrics/recall(B)"])
    print("mAP50:", last_row["metrics/mAP50(B)"])
    print("mAP50-95:", last_row["metrics/mAP50-95(B)"])
