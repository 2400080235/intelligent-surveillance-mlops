
import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# ---------------------------------
# PROJECT SETTINGS
# ---------------------------------

st.set_page_config(
    page_title="Intelligent Surveillance System",
    page_icon="🎥",
    layout="wide"
)

st.title("Intelligent Surveillance System")
st.write("MLOps Pipeline for Intelligent Surveillance System")
st.write("Upload a surveillance video to detect objects using YOLO.")

MODEL_PATH = Path(__file__).parent / "runs" / "train" / "surveillance_demo" / "weights" / "best.pt"


# ---------------------------------
# LOAD YOLO MODEL
# ---------------------------------

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# ---------------------------------
# VIDEO UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload a surveillance video",
    type=["mp4", "avi", "mov", "mkv", "webm"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("Process Video", type="primary"):

        if not MODEL_PATH.exists():
            st.error("Trained YOLO model not found.")
            st.stop()

        with st.spinner("Loading YOLO model..."):
            model = load_model()

        input_path = None
        output_path = None
        writer = None

        try:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            ) as temp_input:
                temp_input.write(uploaded_file.getbuffer())
                input_path = temp_input.name

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("Unable to open the uploaded video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Resize large videos to reduce memory usage
            scale = min(640 / max(width, height), 1.0)
            output_width = max(1, int(width * scale))
            output_height = max(1, int(height * scale))

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_output:
                output_path = temp_output.name

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (output_width, output_height)
            )

            if not writer.isOpened():
                st.error("Unable to create output video.")
                st.stop()

            progress = st.progress(0)
            status = st.empty()

            frame_count = 0
            detections = 0
            class_counts = Counter()

            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame = cv2.resize(
                    frame,
                    (output_width, output_height)
                )

                result = model.predict(
                    frame,
                    imgsz=640,
                    conf=0.4,
                    verbose=False
                )[0]

                annotated_frame = result.plot()
                writer.write(annotated_frame)

                if result.boxes is not None:
                    for class_id in result.boxes.cls.tolist():
                        class_name = model.names[int(class_id)]
                        class_counts[class_name] += 1
                        detections += 1

                frame_count += 1

                if total_frames > 0:
                    progress.progress(
                        min(frame_count / total_frames, 1.0)
                    )

                if frame_count % 10 == 0:
                    status.write(
                        f"Processed {frame_count} frames..."
                    )

            cap.release()
            writer.release()
            writer = None

            progress.progress(1.0)
            status.success("Video processing completed!")

            # Convert to browser-compatible H.264
            import imageio_ffmpeg
            import subprocess

            h264_path = output_path.replace(".mp4", "_h264.mp4")

            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-i", output_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    h264_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            st.subheader("Processing Results")

            col1, col2 = st.columns(2)
            col1.metric("Frames Processed", frame_count)
            col2.metric("Total Detections", detections)

            st.subheader("Detected Objects")

            if class_counts:
                st.table(
                    [
                        {"Object": name, "Detections": count}
                        for name, count in class_counts.items()
                    ]
                )
            else:
                st.info("No objects were detected.")

            st.subheader("Processed Video")
            st.video(h264_path)

            with open(h264_path, "rb") as video_file:
                st.download_button(
                    "Download Processed Video",
                    data=video_file,
                    file_name="surveillance_detected.mp4",
                    mime="video/mp4"
                )

            os.remove(h264_path)

        except Exception as error:
            st.error(f"Video processing failed: {error}")

        finally:
            if writer is not None:
                writer.release()

            if input_path and os.path.exists(input_path):
                os.remove(input_path)

            if output_path and os.path.exists(output_path):
                os.remove(output_path)
