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
        smaller_area = min(first_area, second_area)
        return intersection / union if union else 0.0, intersection / smaller_area if smaller_area else 0.0

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

    def infer(self, image, prompts, threshold):
        prompts = [prompt.strip() for prompt in prompts if prompt and prompt.strip()]
        if not prompts:
            return []

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

        return self._deduplicate(results)
