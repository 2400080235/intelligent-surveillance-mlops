
import cv2
import time
import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Input and output video paths

# Choose which recorded video to process
print("===== VIDEO SELECTION =====")
print("1. Original surveillance video")
print("2. New traffic video (bikes and cars)")

choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    video_path = "data/video/surveillance.mp4"
    output_path = "outputs/original_detected.mp4"
elif choice == "2":
    video_path = "data/video/surveillance2.mp4"
    output_path = "outputs/traffic_detected.mp4"
else:
    print("Invalid choice! Please enter 1 or 2.")
    raise SystemExit(1)

os.makedirs("outputs", exist_ok=True)

# Open the recorded video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    raise SystemExit(1)
print("Video opened successfully!", flush=True)
print("Video FPS:", cap.get(cv2.CAP_PROP_FPS), flush=True)
print("Video width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), flush=True)
print("Video height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT), flush=True)
print("Loading YOLO model and processing frames...", flush=True)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25.0

# Save the annotated video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path, fourcc, fps, (width, height)
)

if not writer.isOpened():
    cap.release()
    print("ERROR: Could not create output video.")
    raise SystemExit(1)

frame_count = 0
total_detections = 0
total_inference_time = 0

print("Recorded video opened successfully!")
print("YOLO11n object detection started.")
print("Press Q to stop early.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    start = time.perf_counter()

    results = model.predict(
        frame,
        imgsz=320,
        conf=0.4,
        verbose=False
    )

    inference_time = time.perf_counter() - start
    detections = len(results[0].boxes)

    annotated_frame = results[0].plot()

    writer.write(annotated_frame)
    cv2.imshow("Recorded Video - YOLO11n Detection", annotated_frame)

    frame_count += 1
    total_detections += detections
    total_inference_time += inference_time

    if frame_count % 30 == 0:
        print(
            f"Frames: {frame_count} | "
            f"Objects in current frame: {detections} | "
            f"Inference: {inference_time * 1000:.2f} ms"
        )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("\n===== VIDEO DETECTION SUMMARY =====")
print("Frames processed:", frame_count)
print("Total detections:", total_detections)

if frame_count > 0:
    print(
        "Average inference time:",
        round(total_inference_time / frame_count * 1000, 2),
        "ms"
    )

print("Annotated video saved to:", output_path)
