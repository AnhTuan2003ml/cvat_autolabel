from threading import Lock

import torch
from ultralytics import YOLO


class ModelHandler:
    labels = ["car", "truck", "bus", "van"]

    def __init__(self):
        self.model = YOLO("best.pt")
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = torch.cuda.is_available()
        self.lock = Lock()

    def infer(self, image, threshold):
        with self.lock:
            prediction = self.model.predict(
                image,
                imgsz=640,
                conf=threshold,
                iou=0.50,
                device=self.device,
                half=self.half,
                verbose=False,
            )[0]

        boxes = prediction.boxes
        if boxes is None:
            return []

        results = []
        for coordinates, confidence, class_id in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
        ):
            index = int(class_id)
            if index >= len(self.labels):
                continue
            results.append(
                {
                    "type": "rectangle",
                    "label": self.labels[index],
                    "points": coordinates,
                    "confidence": float(confidence),
                    "attributes": [],
                }
            )

        return results
