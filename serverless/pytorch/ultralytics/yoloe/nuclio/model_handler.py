from threading import Lock

import torch
from ultralytics import YOLOE


class ModelHandler:
    def __init__(self):
        self.model = YOLOE("yoloe-26x-seg.pt")
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

    def _predict(self, image, prompts, threshold, offset=(0, 0)):
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
                    "points": [
                        coordinates[0] + offset[0],
                        coordinates[1] + offset[1],
                        coordinates[2] + offset[0],
                        coordinates[3] + offset[1],
                    ],
                    "confidence": float(confidence),
                    "attributes": [],
                }
            )
        return results

    @staticmethod
    def _tile_positions(length, tile_size=640, overlap=0.20):
        if length <= tile_size:
            return [0]
        stride = max(1, int(tile_size * (1.0 - overlap)))
        positions = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def infer(self, image, prompts, threshold):
        prompts = [prompt.strip() for prompt in prompts if prompt and prompt.strip()]
        if not prompts:
            return []

        with self.lock:
            if self.prompts != prompts:
                self.model.set_classes(prompts)
                self.prompts = list(prompts)

            results = self._predict(image, prompts, threshold)
            width, height = image.size
            x_positions = self._tile_positions(width)
            y_positions = self._tile_positions(height)
            if len(x_positions) > 1 or len(y_positions) > 1:
                for top in y_positions:
                    for left in x_positions:
                        tile = image.crop((left, top, left + 640, top + 640))
                        results.extend(self._predict(tile, prompts, threshold, (left, top)))

        return self._deduplicate(results)
