import torch
from ultralytics import YOLO

class YOLOv11Trainer:
    def __init__(self, model_path, data_yaml, train_name, device):
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        self.train_name = train_name

    def train(self, epochs=1, imgsz=480):
        
        self.model.train(
            data=self.data_yaml,
            name=self.train_name,
            epochs=epochs,
            imgsz=imgsz,
            device=self.device
        )    