from threading import Lock

import torch
from ultralytics import YOLOE


class ModelHandler:
    def __init__(self):
        self.model = YOLOE("yoloe-26l-seg.pt")
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = torch.cuda.is_available()
        self.prompts = None
        self.lock = Lock()

    def infer(self, image, prompts, threshold):
        with self.lock:
            if self.prompts != prompts:
                self.model.set_classes(prompts)
                self.prompts = list(prompts)

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
            prompt_index = int(class_id)
            if prompt_index >= len(prompts):
                continue
            results.append(
                {
                    "type": "rectangle",
                    "label": prompts[prompt_index],
                    "points": coordinates,
                    "confidence": float(confidence),
                    "attributes": [],
                }
            )

        return results
