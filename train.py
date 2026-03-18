from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

results = model.train(
    data="License-Plate-Recognition-4/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device="mps",
    patience=5,
    save=True,
    save_period=1,
    project="models/runs",
    name="plate_detector",
    pretrained=True,
    verbose=True
)

print("Done! Model saved in models/runs/plate_detector/weights/best.pt")
