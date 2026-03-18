import cv2
import os
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer as ONNXPlateRecognizer
from database import init_db
from alert import check_and_alert

def preprocess_plate(plate_crop):
    plate_crop = cv2.resize(plate_crop, (333, 75))
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

# initialize database
init_db()

# only cache flagged vehicles
flagged_cache = {}  # track_id -> plate_text

# find next available number
counter = 1
while os.path.exists(f"output/lpr_output_{counter}.mp4"):
    counter += 1
output_path = f"output/lpr_output_{counter}.mp4"

# initialize models
vehicle_model = YOLO("models/yolov8l.pt")
plate_model = YOLO("models/plate_detection.pt")
reader = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model')

cap = cv2.VideoCapture("videos/traffic4.mp4")

# video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Processing... output will be saved to {output_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished!")
        break

    # step 1 — detect and track vehicles
    vehicle_results = vehicle_model.track(
        source=frame,
        persist=True,
        classes=[2, 3, 5, 7],
        conf=0.3,
        iou=0.4,
        device="mps",
        tracker="bytetrack.yaml"
    )

    for result in vehicle_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id) if box.id is not None else -1
            label = vehicle_model.names[int(box.cls)]

            plate_text = ""
            is_alert = False

            # if already flagged — use cache, skip detection
            if track_id in flagged_cache:
                plate_text = flagged_cache[track_id]
                is_alert = True
            else:
                # crop vehicle
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue

                # step 2 — detect plate inside vehicle crop
                plate_results = plate_model(
                    vehicle_crop,
                    conf=0.4,
                    device="mps"
                )

                for plate_result in plate_results:
                    for plate_box in plate_result.boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])

                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        if plate_crop.size == 0:
                            continue

                        processed_plate = preprocess_plate(plate_crop)
                        result = reader.run(processed_plate)
                        if result:
                            plate_text = result[0].plate
                            is_alert = check_and_alert(plate_text, label, track_id)

                            # lock only if flagged
                            if is_alert:
                                flagged_cache[track_id] = plate_text
                                print(f"🚨 Flagged vehicle locked #{track_id}: {plate_text}")

                        # draw plate box in red
                        cv2.rectangle(vehicle_crop,
                                    (px1, py1), (px2, py2),
                                    (0, 0, 255), 2)

            # draw vehicle box — red if flagged, green if normal
            box_color = (0, 0, 255) if is_alert else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # draw label
            cv2.putText(frame, f"{label} #{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # draw plate on right side of label
            if plate_text:
                label_text = f"{label} #{track_id} | "
                label_width = cv2.getTextSize(label_text,
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                cv2.putText(frame, plate_text,
                            (x1 + label_width, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # write plate number inside box if flagged
            if is_alert and plate_text:
                text_x = x1 + (x2 - x1) // 2 - 50
                text_y = y1 + (y2 - y1) // 2
                cv2.putText(frame, plate_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Traffic Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved to {output_path}")
