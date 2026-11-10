"""Lazy GPU model loading and inference, using the notebook's model IDs and thresholds."""
from dataclasses import dataclass
from pathlib import Path
import os
import cv2
import numpy as np
from .errors import GPURequiredError, InferenceError, ModelLoadError

DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"


@dataclass
class Detection:
    masks: list[np.ndarray]
    sam_scores: list[float]
    dino_score: float | None


def require_cuda(torch, device):
    if not torch.cuda.is_available():
        raise GPURequiredError(
            "CUDA GPU is required. In Colab select Runtime > Change runtime type > GPU. "
            "On a server install CUDA-enabled PyTorch and expose NVIDIA GPU 0."
        )
    try:
        probe = torch.ones(1, device=device)
        probe.sum().item()
        torch.cuda.synchronize()
        del probe
    except Exception as error:
        raise GPURequiredError("CUDA was detected but a GPU operation failed") from error


class TorchBackend:
    def __init__(self, config):
        try:
            import torch
        except ImportError as error:
            raise ModelLoadError("Install the package dependencies, including CUDA-enabled PyTorch") from error
        require_cuda(torch, config.device)
        self.torch = torch
        self.device = config.device
        self.config = config
        cache = Path(config.cache_dir or os.environ.get("CARDID_CACHE_DIR", Path.home()/".cache"/"cardid"))
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamProcessor, SamModel
            import easyocr
            cache.mkdir(parents=True, exist_ok=True)
            options = {"cache_dir": str(cache/"huggingface"), "local_files_only": config.offline}
            self.dino_processor = AutoProcessor.from_pretrained(DINO_ID, **options)
            self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                DINO_ID, use_safetensors=True, **options).to(self.device).eval()
            self.sam_processor = SamProcessor.from_pretrained(SAM_ID, **options)
            self.sam = SamModel.from_pretrained(
                SAM_ID, use_safetensors=True, **options).to(self.device).eval()
            self.reader = easyocr.Reader(
                ['fa', 'en'], gpu=self.device,
                model_storage_directory=str(cache/"easyocr"),
                user_network_directory=str(cache/"easyocr_network"),
                download_enabled=not config.offline, verbose=False,
            )
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self.face_detector.empty():
                raise ModelLoadError("OpenCV face cascade missing; install the specified OpenCV 4.x dependency")
            self.model_devices = {}
            for name, model in [("dino", self.dino), ("sam", self.sam),
                                ("orientation_detector", self.reader.detector),
                                ("orientation_recognizer", self.reader.recognizer)]:
                actual = next(model.parameters()).device
                if actual.type != "cuda" or actual.index not in (None, 0):
                    raise GPURequiredError(f"{name} is on {actual}; CUDA GPU 0 is required")
                self.model_devices[name] = str(actual)
            self.info = {"device": self.device, "gpu_name": torch.cuda.get_device_name(0),
                         "model_devices": self.model_devices, "dino_model": DINO_ID,
                         "sam_model": SAM_ID, "orientation_model": "easyocr:fa,en"}
        except (GPURequiredError, ModelLoadError):
            raise
        except Exception as error:
            raise ModelLoadError(
                "Could not load extraction models. First use needs internet for weights; "
                "offline=True requires a populated cache. Use an isolated environment "
                "with this package's dependencies. Original error: " + str(error)
            ) from error

    def detect_masks(self, pil):
        """Return SAM candidates for the highest-scoring DINO card detection."""
        return self._detect_masks(pil, alternatives=False)

    def detect_alternative_masks(self, pil):
        """Bounded recovery only: segment the next two DINO detections."""
        return self._detect_masks(pil, alternatives=True)

    def _detect_masks(self, pil, *, alternatives):
        torch = self.torch
        width, height = pil.size
        try:
            inputs = self.dino_processor(
                images=pil, text="Iranian national identity card. identity card.",
                return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.dino(**inputs)
            detection = self.dino_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold, target_sizes=[(height, width)])[0]
            if len(detection['boxes']) == 0:
                return Detection([], [], None)
            best = int(detection['scores'].argmax())
            order = detection['scores'].argsort(descending=True).detach().cpu().tolist()
            indices = [index for index in order if index != best][:2] if alternatives else [best]
            if len(indices) == 0:
                return Detection([], [], None)
            boxes = detection['boxes'][indices].detach().cpu().tolist()
            score = float(detection['scores'][indices[0]])
            sam_inputs = self.sam_processor(images=pil, input_boxes=[boxes], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                sam_outputs = self.sam(**sam_inputs, multimask_output=True)
            masks = self.sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(), sam_inputs['original_sizes'].cpu(),
                sam_inputs['reshaped_input_sizes'].cpu())[0]
            masks = masks.reshape(-1, height, width).numpy() > 0
            scores = sam_outputs.iou_scores.detach().cpu().numpy().reshape(-1)
            return Detection([m.astype(np.uint8) for m in masks],
                             [float(value) for value in scores], score)
        except torch.cuda.OutOfMemoryError as error:
            raise InferenceError(
                "GPU memory exhausted. Reuse one extractor per process and avoid concurrent "
                "GPU workers or excessively large input photos."
            ) from error
        except Exception as error:
            raise InferenceError(f"Card detection/segmentation failed: {error}") from error
