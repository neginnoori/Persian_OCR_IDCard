from threading import RLock
from time import perf_counter
import cv2
import numpy as np
from PIL import Image
from .backend import TorchBackend
from .errors import CardNotFoundError, InferenceError, ReviewRequiredError
from .geometry import quad_from_mask, warp_candidates, choose_orientation
from .images import normalize_image
from .types import ExtractionResult, ExtractorConfig


class CardExtractor:
    """Reusable extractor. One instance serializes calls and loads models once.

    extract() returns a PIL RGB card on success. extract_result() exposes the candidate
    and diagnostics for uncertain/no-card inputs. No image is saved or displayed.
    Input NumPy arrays default to RGB; explicitly use color_order='BGR' for OpenCV.
    """

    def __init__(self, config: ExtractorConfig | None = None):
        self.config = config or ExtractorConfig()
        self._backend = None
        self._lock = RLock()

    def load(self):
        """Load models on CUDA now (optional startup warmup); return self."""
        with self._lock:
            if self._backend is None:
                self._backend = TorchBackend(self.config)
        return self

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    @property
    def runtime_info(self) -> dict:
        with self._lock:
            return dict(self._backend.info) if self._backend is not None else {"loaded": False}

    def extract(self, image, *, color_order="RGB", orientation_override=None) -> Image.Image:
        """Return only the corrected card image, or raise an explicit extraction error."""
        result = self.extract_result(image, color_order=color_order, orientation_override=orientation_override)
        if result.status == "not_found":
            raise CardNotFoundError(result.reason or "No card found")
        if result.status != "ok":
            raise ReviewRequiredError(result)
        return result.image

    def extract_result(self, image, *, color_order="RGB", orientation_override=None) -> ExtractionResult:
        """Process one photo; status is ok, needs_review or not_found.

        orientation_override is the notebook's cyclic corner shift: 0/90/180/270.
        It is relative to the geometric base warp, not the raw camera rotation.
        It does not override unreliable geometry.
        """
        if orientation_override is not None and orientation_override not in (0,90,180,270):
            raise ValueError("orientation_override must be 0, 90, 180, 270, or None")
        started = perf_counter()
        pil = normalize_image(image, color_order=color_order)
        with self._lock:
            self.load()
            backend = self._backend
            detection = backend.detect_masks(pil)
            metadata = {"original_size": list(pil.size), "dino_score": detection.dino_score,
                        "coordinate_space": "exif_corrected_input", "runtime": self.runtime_info}
            if not detection.masks:
                metadata['elapsed_ms'] = (perf_counter()-started)*1000
                return ExtractionResult("not_found", reason="no_card_detection", metadata=metadata)
            proposals = []
            for mask, sam_score in zip(detection.masks, detection.sam_scores):
                try:
                    quad, metrics = quad_from_mask(mask, min_iou=self.config.min_quad_iou)
                except ValueError:
                    continue
                metrics['sam_predicted_iou'] = sam_score
                proposals.append((quad, metrics))
            if not proposals:
                metadata['elapsed_ms'] = (perf_counter()-started)*1000
                return ExtractionResult("needs_review", reason="no_reliable_quadrilateral", metadata=metadata)
            quad, metrics = max(proposals, key=lambda item: (
                item[1]['geometry_ok'], item[1]['mask_quad_iou']+.05*item[1]['sam_predicted_iou']))
            metadata.update(metrics)
            bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            try:
                candidates = warp_candidates(bgr, quad, self.config.width, self.config.height)
                chosen, orientation = choose_orientation(
                    candidates, backend.reader, backend.face_detector, override=orientation_override,
                    min_margin=self.config.min_orientation_margin)
            except Exception as error:
                # Runtime/model errors must be visible to the service, not silently accepted.
                raise InferenceError(f"Card rectification/orientation failed: {error}") from error
            metadata.update(orientation)
            matrix = chosen['H']
            metadata.update(original_to_card_H=matrix.tolist(), card_to_original_H=np.linalg.inv(matrix).tolist(),
                            source_corners=chosen['quad'].tolist(), output_size=[self.config.width,self.config.height],
                            elapsed_ms=(perf_counter()-started)*1000)
            output = Image.fromarray(cv2.cvtColor(chosen['card'], cv2.COLOR_BGR2RGB))
            reasons = []
            if not metrics['geometry_ok']:
                reasons.append("unreliable_geometry")
            if not orientation['orientation_ok']:
                reasons.append("ambiguous_orientation")
            return ExtractionResult("needs_review" if reasons else "ok", image=output,
                                    reason=",".join(reasons) or None, metadata=metadata)
