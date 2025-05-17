from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

model.train(data="model/config.yaml", epochs=30)

print("Training continued successfully!")