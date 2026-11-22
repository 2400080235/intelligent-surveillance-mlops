
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import uuid

import cv2
import imageio_ffmpeg

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    Response,
)

from werkzeug.utils import secure_filename
from ultralytics import YOLO

from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "runs" / "train" / "surveillance_demo"
    / "weights" / "best.pt"
)

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
HISTORY_FILE = BASE_DIR / "video_history.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# --------------------------------------------------
# 2. PROMETHEUS MONITORING METRICS
# --------------------------------------------------

VIDEO_UPLOADS = Counter(
    "surveillance_video_uploads_total",
    "Total number of uploaded videos"
)

VIDEOS_PROCESSED = Counter(
    "surveillance_videos_processed_total",
    "Total number of successfully processed videos"
)

DETECTIONS_TOTAL = Counter(
    "surveillance_detections_total",
    "Total number of objects detected across processed video frames"
)

PROCESSING_ERRORS = Counter(
    "surveillance_processing_errors_total",
    "Total number of video processing errors"
)

LAST_VIDEO_FRAMES = Gauge(
    "surveillance_last_video_frames",
    "Number of frames in the most recently processed video"
)

LAST_PROCESSING_STATUS = Gauge(
    "surveillance_last_processing_success",
    "1 if the latest processing attempt succeeded, otherwise 0"
)


# --------------------------------------------------
# 3. LOAD AND SAVE VIDEO HISTORY
# --------------------------------------------------

def load_video_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)

        if isinstance(history, list):
            return history

        app.logger.error("Video history file has an invalid format.")
        return []

    except (json.JSONDecodeError, OSError):
        app.logger.exception("Could not read video history.")
        return []


def save_video_history(history):
    temporary_file = HISTORY_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    temporary_file.replace(HISTORY_FILE)


# --------------------------------------------------
# 4. LOAD TRAINED YOLO MODEL
# --------------------------------------------------

print("Loading trained YOLO model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully!")


# --------------------------------------------------
# 5. CHECK VIDEO FILE TYPE
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# 6. PROCESS VIDEO USING YOLO
# --------------------------------------------------

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    frames = 0
    detections = 0
    objects = {}

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            predictions = model.predict(
                frame,
                conf=0.25,
                verbose=False,
            )

            prediction = predictions[0]
            annotated_frame = prediction.plot()

            writer.write(annotated_frame)
            frames += 1

            if prediction.boxes is not None:
                for class_id in prediction.boxes.cls.tolist():
                    class_id = int(class_id)

                    class_name = model.names.get(
                        class_id, str(class_id)
                    )

                    objects[class_name] = (
                        objects.get(class_name, 0) + 1
                    )

                    detections += 1

            if frames % 30 == 0:
                print(f"Processed {frames} frames...")

    finally:
        cap.release()
        writer.release()

    if frames == 0:
        raise RuntimeError("No video frames were processed.")

    print("Video detection completed.")
    print("Frames processed:", frames)
    print("Total detections:", detections)
    print("Detected categories:", objects)

    return frames, detections, objects


# --------------------------------------------------
# 7. CONVERT VIDEO TO H.264 FOR CHROME
# --------------------------------------------------

def convert_to_h264(input_path, output_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "H.264 conversion failed:\n"
            + completed.stderr[-3000:]
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Converted video is missing or empty.")

    print("H.264 conversion successful:", output_path.name)


# --------------------------------------------------
# 8. DASHBOARD AND VIDEO UPLOAD
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("video")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a video to upload."

        elif not allowed_file(uploaded_file.filename):
            error = (
                "Unsupported file type. Please upload "
                "MP4, AVI, MOV, MKV, or WEBM."
            )

        else:
            safe_name = secure_filename(uploaded_file.filename)
            unique_id = uuid.uuid4().hex[:10]

            input_path = UPLOAD_DIR / f"{unique_id}_{safe_name}"

            raw_output_path = (
                OUTPUT_DIR / f"{unique_id}_detected.mp4"
            )

            h264_output_path = (
                OUTPUT_DIR / f"{unique_id}_detected_h264.mp4"
            )

            VIDEO_UPLOADS.inc()
            LAST_PROCESSING_STATUS.set(0)

            try:
                uploaded_file.save(str(input_path))

                print("\nUploaded:", safe_name)
                print("Starting YOLO detection...")

                frames, detections, objects = process_video(
                    input_path,
                    raw_output_path,
                )

                print("Converting video for browser playback...")

                convert_to_h264(
                    raw_output_path,
                    h264_output_path,
                )

                # Create the result record.
                result = {
                    "video": h264_output_path.name,
                    "original_name": safe_name,
                    "frames": frames,
                    "total_detections": detections,
                    "objects": objects,
                    "processed_at": datetime.now().strftime(
                        "%d-%m-%Y %I:%M:%S %p"
                    ),
                }

                # Save the successful result in persistent history.
                history = load_video_history()
                history.insert(0, result)
                save_video_history(history)

                # Update Prometheus metrics.
                VIDEOS_PROCESSED.inc()
                DETECTIONS_TOTAL.inc(detections)
                LAST_VIDEO_FRAMES.set(frames)
                LAST_PROCESSING_STATUS.set(1)

                print("Dashboard results ready!")
                print("Processed video:", result["video"])
                print("Video history saved successfully!")

            except Exception as exc:
                app.logger.exception("Video processing failed")
                PROCESSING_ERRORS.inc()
                LAST_PROCESSING_STATUS.set(0)
                result = None
                error = f"Video processing failed: {exc}"

    # Load all previous successful video records.
    history = load_video_history()

    return render_template(
        "index.html",
        result=result,
        history=history,
        error=error,
    )


# --------------------------------------------------
# 9. PROMETHEUS METRICS ENDPOINT
# --------------------------------------------------

@app.route("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST,
    )


# --------------------------------------------------
# 10. SERVE PROCESSED VIDEO
# --------------------------------------------------

@app.route("/outputs/<path:filename>")
def output_video(filename):
    return send_from_directory(
        str(OUTPUT_DIR),
        filename,
        mimetype="video/mp4",
        conditional=True,
    )


# --------------------------------------------------
# 11. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    print("\nStarting Intelligent Surveillance Dashboard...")
    print("Open http://127.0.0.1:5001 in Chrome.\n")

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5001)),
        debug=False,
    )
