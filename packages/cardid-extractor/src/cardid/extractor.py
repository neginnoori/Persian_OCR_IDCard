from threading import RLock
from time import perf_counter
import cv2
import numpy as np
from PIL import Image
from .backend import TorchBackend
from .errors import CardNotFoundError, InferenceError, ReviewRequiredError
from .geometry import card_edge_quads, card_frame_quads, choose_orientation, quad_from_mask, quad_iou, warp_candidates
from .images import normalize_image
from .types import ExtractionResult, ExtractorConfig


_MIN_REVIEW_AREA = .15



def _orientation_evidence(orientation):
    scores = orientation['orientation_scores']
    best = scores[orientation['corner_shift_degrees'] // 90]
    unique_left_face = best['face_on_left'] and sum(score['face_on_left'] for score in scores) == 1
    return best, unique_left_face


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
            proposals = []
            dino_scores = detection.mask_dino_scores or [detection.dino_score] * len(detection.masks)
            for mask, sam_score, dino_score in zip(detection.masks, detection.sam_scores, dino_scores):
                try:
                    quad, metrics = quad_from_mask(mask, min_iou=self.config.min_quad_iou)
                except ValueError:
                    continue
                metrics['sam_predicted_iou'] = sam_score
                metrics['dino_score'] = dino_score
                metrics['proposal_source'] = 'dino_sam'
                proposals.append((quad, metrics))
            proposals.sort(key=lambda item: (
                item[1]['geometry_ok'], item[1]['dino_score'] or 0,
                item[1]['mask_quad_iou']+.05*item[1]['sam_predicted_iou']), reverse=True)
            bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            try:
                selected = self._select_card(bgr, proposals, backend, orientation_override)
            except Exception as error:
                # Runtime/model errors must be visible to the service, not silently accepted.
                raise InferenceError(f"Card rectification/orientation failed: {error}") from error
            if selected is None:
                metadata['elapsed_ms'] = (perf_counter()-started)*1000
                if not detection.masks:
                    return ExtractionResult("not_found", reason="no_card_detection", metadata=metadata)
                return ExtractionResult("needs_review", reason="no_reliable_quadrilateral", metadata=metadata)
            chosen, orientation, metrics = selected
            metadata.update(metrics)
            metadata.update(orientation)
            if (not orientation['orientation_ok']
                    and cv2.contourArea(chosen['quad']) < _MIN_REVIEW_AREA * pil.width * pil.height):
                metadata['elapsed_ms'] = (perf_counter()-started)*1000
                return ExtractionResult("needs_review", reason="no_card_layout_evidence", metadata=metadata)
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

    def _select_card(self, bgr, proposals, backend, override):
        review = None
        checked = []
        for quad, metrics in proposals:
            if any(quad_iou(quad, other) >= .88 for other in checked):
                continue
            if not metrics['geometry_ok'] and review is not None:
                break
            checked.append(quad)
            chosen, orientation = self._orient(bgr, quad, backend, override)
            if metrics['geometry_ok'] and orientation['orientation_ok']:
                return chosen, orientation, metrics
            if review is None or orientation['orientation_ok']:
                review = chosen, orientation, metrics

        height, width = bgr.shape[:2]
        edge_review = None
        for quad in card_edge_quads(bgr):
            # An outline alone is insufficient to identify a card.
            chosen, orientation = self._orient(bgr, quad, backend, None)
            touches_border = bool(np.any(quad.min(axis=0) <= 1) or
                                  np.any(quad.max(axis=0) >= [width - 2, height - 2]))
            metrics = {
                'geometry_ok': not touches_border,
                'touches_image_border': touches_border,
                'quad_method': 'card_edges', 'proposal_source': 'card_edges',
                'dino_score': None,
            }
            if orientation['orientation_ok'] and metrics['geometry_ok']:
                if override is not None:
                    chosen, orientation = self._orient(bgr, quad, backend, override)
                return chosen, orientation, metrics

            best, unique_left_face = _orientation_evidence(orientation)
            has_layout = orientation['orientation_ok'] or best['field_row_count'] >= 2 or unique_left_face
            if edge_review is None and has_layout and cv2.contourArea(quad) >= _MIN_REVIEW_AREA * width * height:
                edge_review = chosen, orientation, metrics

        if review is not None and cv2.contourArea(review[0]['quad']) >= _MIN_REVIEW_AREA * width * height:
            return review
        if edge_review is not None:
            return edge_review

        for quad in card_frame_quads(bgr):
            chosen, orientation = self._orient(bgr, quad, backend, None)
            best, unique_left_face = _orientation_evidence(orientation)
            if orientation['orientation_ok'] or (unique_left_face and best['right_text_row_count'] >= 2):
                if override is not None:
                    chosen, orientation = self._orient(bgr, quad, backend, override)
                return chosen, orientation, {
                    'geometry_ok': False, 'touches_image_border': True,
                    'quad_method': 'image_frame', 'proposal_source': 'image_frame',
                    'dino_score': None,
                }
        return review

    def _orient(self, bgr, quad, backend, override):
        candidates = warp_candidates(bgr, quad, self.config.width, self.config.height)
        return choose_orientation(
            candidates, backend.reader, backend.face_detector, override=override,
            min_margin=self.config.min_orientation_margin)
