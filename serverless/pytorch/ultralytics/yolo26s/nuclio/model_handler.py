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

    @staticmethod
    def _overlap(first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        iou = intersection / union if union else 0.0
        smaller_area = min(first_area, second_area)
        containment = intersection / smaller_area if smaller_area else 0.0
        return iou, containment

    @classmethod
    def _deduplicate(cls, results):
        kept = []
        for result in sorted(results, key=lambda item: item["confidence"], reverse=True):
            duplicate = any(
                cls._overlap(result["points"], previous["points"])[0] >= 0.65
                or cls._overlap(result["points"], previous["points"])[1] >= 0.9
                for previous in kept
            )
            if not duplicate:
                kept.append(result)
        return kept

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

        return self._deduplicate(results)
