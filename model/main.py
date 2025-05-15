from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

results = model.train(data='model/config.yaml', epochs=30)