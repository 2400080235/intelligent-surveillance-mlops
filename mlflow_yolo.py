import cv2
import time
import mlflow
from ultralytics import YOLO

# -----------------------------
# MLflow configuration
# -----------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Intelligent Surveillance System")

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolo11n.pt")

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully.")
print("YOLO + MLflow is running.")
print("Press Q to stop.")

# Variables for real measurements
frame_count = 0
total_detections = 0
total_inference_time = 0

# -----------------------------
# Start MLflow run
# -----------------------------
with mlflow.start_run() as run:

    # Log project parameters
    mlflow.log_param("model", "YOLO11n")
    mlflow.log_param("dataset", "COCO128")
    mlflow.log_param("image_size", 320)
    mlflow.log_param("confidence_threshold", 0.4)
    mlflow.log_param("input", "webcam")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        # Measure REAL YOLO inference time
        start_time = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=320,
            conf=0.4,
            verbose=False
        )

        inference_time = time.perf_counter() - start_time

        # Count detected objects
        detections = 0

        if results[0].boxes is not None:
            detections = len(results[0].boxes)

        frame_count += 1
        total_detections += detections
        total_inference_time += inference_time

        # Display YOLO detections
        annotated_frame = results[0].plot()

        cv2.imshow("Real-Time YOLO + MLflow", annotated_frame)

        # Stop with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # -----------------------------
    # Calculate actual metrics
    # -----------------------------
    if frame_count > 0:

        average_inference_time_ms = (
            total_inference_time / frame_count
        ) * 1000

        average_detections_per_frame = (
            total_detections / frame_count
        )

        # Log REAL metrics to MLflow
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
        print("MLflow run completed successfully!")
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

# -----------------------------
# Release camera
# -----------------------------
cap.release()
cv2.destroyAllWindows()