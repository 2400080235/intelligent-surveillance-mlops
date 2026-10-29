
from pathlib import Path
import subprocess
import uuid

import cv2
import imageio_ffmpeg
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "train"
    / "surveillance_demo"
    / "weights"
    / "best.pt"
)

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# --------------------------------------------------
# 2. LOAD TRAINED YOLO MODEL
# --------------------------------------------------

print("Loading trained YOLO model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully!")


# --------------------------------------------------
# 3. CHECK VIDEO FILE TYPE
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# 4. PROCESS VIDEO USING YOLO
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

            # Run YOLO object detection.
            predictions = model.predict(
                frame,
                conf=0.25,
                verbose=False,
            )

            prediction = predictions[0]

            # Draw boxes and labels on the frame.
            annotated_frame = prediction.plot()

            writer.write(annotated_frame)

            frames += 1

            # Count detections across frames.
            if prediction.boxes is not None:
                for class_id in prediction.boxes.cls.tolist():
                    class_id = int(class_id)
                    class_name = model.names.get(
                        class_id,
                        str(class_id),
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
# 5. CONVERT VIDEO TO H.264 FOR CHROME
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
# 6. DASHBOARD AND VIDEO UPLOAD
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

            input_path = (
                UPLOAD_DIR / f"{unique_id}_{safe_name}"
            )

            raw_output_path = (
                OUTPUT_DIR / f"{unique_id}_detected.mp4"
            )

            h264_output_path = (
                OUTPUT_DIR / f"{unique_id}_detected_h264.mp4"
            )

            try:
                # Save uploaded video.
                uploaded_file.save(str(input_path))

                print("\nUploaded:", safe_name)
                print("Starting YOLO detection...")

                # Process video and collect results.
                frames, detections, objects = process_video(
                    input_path,
                    raw_output_path,
                )

                # Convert annotated video to browser-compatible MP4.
                print("Converting video for browser playback...")

                convert_to_h264(
                    raw_output_path,
                    h264_output_path,
                )

                # Dashboard results.
                result = {
                    "video": h264_output_path.name,
                    "original_name": safe_name,
                    "frames": frames,
                    "total_detections": detections,
                    "objects": objects,
                }

                print("Dashboard results ready!")
                print("Processed video:", result["video"])

            except Exception as exc:
                app.logger.exception("Video processing failed")
                error = f"Video processing failed: {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
    )


# --------------------------------------------------
# 7. SERVE PROCESSED VIDEO
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
# 8. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    print("\nStarting Intelligent Surveillance Dashboard...")
    print("Open http://127.0.0.1:5001 in Chrome.\n")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
    )
