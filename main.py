from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Camera opened successfully.")
print("Press Q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Run YOLO detection
    results = model.predict(frame, verbose=False)

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Show the video
    cv2.imshow("Real-Time Video Analytics", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()