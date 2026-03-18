import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8l.pt")  # default weights, no training needed

cap = cv2.VideoCapture("videos/traffic1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        source=frame,
        persist=True,
        classes=[2, 3, 5, 7],
        conf=0.3,
        iou=0.4,
        device="mps",
        tracker="bytetrack.yaml"
    )

    annotated_frame = results[0].plot()
    cv2.imshow("Traffic Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
