from ultralytics import YOLO
import os
print(os.getcwd())

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="datasets/garbage/litter.v1i.yolov8/test/images",
    save=True
)

print("Prediction completed!")