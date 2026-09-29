import cv2
import time
import mlflow
from ultralytics import YOLO

# -----------------------------
# MLflow configuration
# -----------------------------
mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment("Intelligent Surveillance System - Docker")

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolo11n.pt")

# -----------------------------
# Open surveillance video
# -----------------------------
video_path = "data/video/surveillance.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open surveillance video.")
    exit()

print("Video opened successfully.")
print("YOLO + MLflow Docker pipeline is running.")

# -----------------------------
# Variables for metrics
# -----------------------------
frame_count = 0
total_detections = 0
total_inference_time = 0

# -----------------------------
# Start MLflow run
# -----------------------------
with mlflow.start_run() as run:

    # Log parameters
    mlflow.log_param("model", "YOLO11n")
    mlflow.log_param("input", "surveillance.mp4")
    mlflow.log_param("environment", "Docker")
    mlflow.log_param("image_size", 320)
    mlflow.log_param("confidence_threshold", 0.4)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Measure YOLO inference time
        start_time = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=320,
            conf=0.4,
            verbose=False
        )

        inference_time = time.perf_counter() - start_time

        # Count detected objects
        detections = len(results[0].boxes)

        # Update metrics
        frame_count += 1
        total_detections += detections
        total_inference_time += inference_time

        # Print progress
        print(
            f"Frame {frame_count} | "
            f"Objects detected: {detections}"
        )

    # -----------------------------
    # Calculate metrics
    # -----------------------------
    if frame_count > 0:

        average_inference_time_ms = (
            total_inference_time / frame_count
        ) * 1000

        average_detections_per_frame = (
            total_detections / frame_count
        )

        # Log metrics to MLflow
        mlflow.log_metric(
            "frames_processed",
            frame_count
        )

        mlflow.log_metric(
            "total_detections",
            total_detections
        )

        mlflow.log_metric(
            "average_inference_time_ms",
            average_inference_time_ms
        )

        mlflow.log_metric(
            "average_detections_per_frame",
            average_detections_per_frame
        )

        print()
        print("================================")
        print("MLflow Docker Run Completed")
        print("================================")
        print("Run ID:", run.info.run_id)
        print("Frames processed:", frame_count)
        print("Total detections:", total_detections)
        print(
            "Average inference time:",
            round(average_inference_time_ms, 2),
            "ms"
        )
        print(
            "Average detections per frame:",
            round(average_detections_per_frame, 2)
        )

cap.release()

print("Video processing completed.")