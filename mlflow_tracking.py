import mlflow
import time

# Connect Python to the MLflow Tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Select our experiment
mlflow.set_experiment("Real-Time Video Analytics")

# Start an MLflow run
with mlflow.start_run() as run:

    model_name = "YOLO11n"
    dataset_name = "COCO128"
    image_size = 320
    confidence_threshold = 0.4

    # Log parameters
    mlflow.log_param("model", model_name)
    mlflow.log_param("dataset", dataset_name)
    mlflow.log_param("image_size", image_size)
    mlflow.log_param("confidence_threshold", confidence_threshold)

    # Temporary performance test
    start_time = time.time()

    time.sleep(1)

    inference_time = time.time() - start_time

    # Log metric
    mlflow.log_metric(
        "inference_time_seconds",
        inference_time
    )

    print("MLflow experiment completed successfully!")
    print("Run ID:", run.info.run_id)
    print("Model:", model_name)
    print("Dataset:", dataset_name)
    print("Inference time:", round(inference_time, 4), "seconds")