
from ultralytics import YOLO

# start from pretrained weights, don't train from scratch
model = YOLO("models/yolov8l.pt")

results = model.train(
    data="datasets/indian_traffic/data.yaml",
    epochs=50,           # number of training rounds
    imgsz=640,
    batch=8,             # lower if you run out of memory
    device="mps",        # apple silicon gpu
    patience=10,         # stops early if no improvement
    save=True,
    project="models/runs",
    name="indian_traffic_v1"
)
