"""Lazy model adapters; importing the library never imports Paddle or Torch."""
import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .errors import InferenceError, ModelLoadError
from .geometry import TextBox


class Detector(Protocol):
    def detect(self, image: Image.Image) -> list[TextBox]: ...


class Recognizer(Protocol):
    def recognize(self, image: Image.Image) -> str: ...


class PaddleDetector:
    def __init__(self, config):
        self.config = config
        self._model = None

    def detect(self, image):
        if self._model is None:
            try:
                from paddleocr import TextDetection
                if self.config.offline and not self.config.detector_model_dir:
                    raise ValueError("offline=True requires detector_model_dir for Paddle weights")
                self._model = TextDetection(model_name=self.config.detector_model,
                                            model_dir=self.config.detector_model_dir,
                                            device=self.config.detector_device, enable_mkldnn=False)
            except Exception as exc:
                raise ModelLoadError(f"Could not load Paddle TextDetection. Install [models]. {exc}") from exc
        try:
            predictions = list(self._model.predict(
                input=cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), batch_size=1))
            if len(predictions) != 1:
                raise ValueError("Expected exactly one result per image")
            raw = predictions[0].json
            data = json.loads(raw) if isinstance(raw, str) else raw
            data = data.get("res", data)
            polygons, scores = data["dt_polys"], data["dt_scores"]
            if len(polygons) != len(scores):
                raise ValueError("Paddle polygons and scores are not aligned")
            result = []
            for index, (polygon, score) in enumerate(zip(polygons, scores), 1):
                box = TextBox(index, tuple(tuple(float(v) for v in p) for p in polygon), float(score))
                if box.score >= self.config.detection_min_score:
                    result.append(box)
            # IDs reflect Paddle's native order and are stable within this one prediction.
            return result
        except Exception as exc:
            raise InferenceError(f"Paddle detection failed: {exc}") from exc


class HezarRecognizer:
    def __init__(self, config):
        self.config = config
        self._model = None
        self._torch = None
        self.info = {"model": config.recognizer_model, "loaded": False}

    def _load(self):
        try:
            import torch
            from hezar.models import Model
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(
                self.config.recognizer_model, revision=self.config.recognizer_revision,
                allow_patterns=["*.yaml", "*.json", "*.pt"],
                cache_dir=str(Path(self.config.cache_dir).expanduser()) if self.config.cache_dir else None,
                local_files_only=self.config.offline,
            )
            if self.config.recognizer_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; select recognizer_device='cpu'")
            model = Model.load(snapshot, device=self.config.recognizer_device).eval()
            self._torch, self._model = torch, model
            self.info = {"model": self.config.recognizer_model, "loaded": True,
                         "requested_revision": self.config.recognizer_revision,
                         "resolved_snapshot": Path(snapshot).name,
                         "reverse_output_digits": bool(model.config.reverse_output_digits),
                         "device": self.config.recognizer_device}
        except Exception as exc:
            raise ModelLoadError(f"Could not load Hezar. Install [models] and populate cache. {exc}") from exc

    def recognize(self, image):
        if self._model is None:
            self._load()
        try:
            with self._torch.inference_mode():
                output = self._model.predict([image], device=self.config.recognizer_device)
            if not isinstance(output, list) or len(output) != 1:
                raise ValueError("Expected one Hezar result for one field crop")
            value = output[0].to_dict() if hasattr(output[0], "to_dict") else output[0]
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise TypeError("Hezar output lacks a text string")
            return value["text"]
        except Exception as exc:
            raise InferenceError(f"Hezar recognition failed: {exc}") from exc
