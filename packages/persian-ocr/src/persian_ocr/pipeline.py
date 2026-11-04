"""Orchestrate the v13 pipeline without notebook globals or implicit output files."""
from collections.abc import Mapping
from dataclasses import asdict
from threading import RLock
from time import perf_counter

from .backends import HezarRecognizer, PaddleDetector
from .config import OCRConfig
from .errors import InferenceError, InvalidImageError, ModelLoadError
from .geometry import crop_field, pixel_zone, select_boxes
from .images import load_image
from .quality import assess, correct_card, sharpen_crop
from .recognition import decide, interpret
from .result import OCRResult


class PersianOCR:
    """One reusable engine. Direct read() input is an aligned, upright card image.

    read_photo() explicitly delegates raw-photo geometry to CardID. Models load lazily.
    Inject detector/recognizer implementations for tests or alternative deployments.
    """

    def __init__(self, config: OCRConfig | None = None, *, detector=None, recognizer=None,
                 extractor=None):
        self.config = config or OCRConfig()
        self.detector = detector if detector is not None else PaddleDetector(self.config)
        self.recognizer = recognizer if recognizer is not None else HezarRecognizer(self.config)
        self.extractor = extractor
        self._lock = RLock()

    def _empty_fields(self, reason):
        return {spec.key: {"label": spec.label, "kind": spec.kind, "value": None,
                           "status": "skipped", "reason": reason, "candidates": {}}
                for spec in self.config.fields}

    def read_extraction(self, extraction, *, approve_review=True, box_overrides=None):
        """Read ok/review card images; reject no-card results before any OCR work.

        approve_review=False restores strict skipping of needs_review inputs.
        Processing a review image preserves its original warning in the result.
        """
        if extraction is None:
            status, image, reason, metadata = "not_found", None, "no_card_found", {}
        elif isinstance(extraction, Mapping):
            status = extraction.get("status")
            image, reason = extraction.get("image"), extraction.get("reason")
            metadata = {k: v for k, v in extraction.items() if k != "image"}
        else:
            status = getattr(extraction, "status", None)
            image, reason = getattr(extraction, "image", None), getattr(extraction, "reason", None)
            metadata = extraction.to_dict() if hasattr(extraction, "to_dict") else {"status": status, "reason": reason}
        status = str(getattr(status, "value", status)).strip().lower().replace(" ", "_")
        source = {"kind": "cardid_extraction", "extraction": metadata,
                  "review_input_processed": bool(status == "needs_review" and approve_review
                                                 and image is not None)}
        if status != "ok" and not (status == "needs_review" and approve_review):
            output_status = "not_found" if status in (
                "not_found", "no_card_found", "no_photos_found", "no_photo_found"
            ) else "needs_review"
            return OCRResult(output_status, self._empty_fields(reason or status), source=source,
                             diagnostics={"ocr_executed": False, "reason": reason or status})
        if image is None:
            if status == "needs_review":
                return OCRResult("needs_review", self._empty_fields("extraction_image_missing"),
                                 source=source, diagnostics={"ocr_executed": False,
                                                            "reason": "extraction_image_missing"})
            raise InvalidImageError("Accepted extraction has no image; pass ExtractionResult, not its pixel-free to_dict()")
        result = self._read_image(image, "RGB", box_overrides, source)
        if status == "needs_review":
            result.diagnostics.update(extraction_needs_review=True, extraction_reason=reason,
                                      ocr_status_before_extraction_review=result.status)
            if result.status != "error":
                result.status = "needs_review"
        return result

    def read_photo(self, image, *, color_order="RGB", extractor=None, approve_review=True,
                   box_overrides=None):
        """Extract a raw camera photo, then gate its status and run OCR on the card only."""
        with self._lock:
            selected = extractor if extractor is not None else self.extractor
            if selected is None:
                try:
                    from cardid import CardExtractor
                except ImportError as exc:
                    raise ModelLoadError("Raw photos require cardid-extractor; install its ZIP or pass an extractor") from exc
                selected = self.extractor = CardExtractor()
            normalized = load_image(image, color_order=color_order)
            extracted = selected.extract_result(normalized)
            return self.read_extraction(extracted, approve_review=approve_review,
                                        box_overrides=box_overrides)

    def read(self, image, *, color_order="RGB", box_overrides=None):
        """Read an aligned card, or accept a CardID status/image result directly."""
        if image is None or isinstance(image, Mapping) or hasattr(image, "status"):
            return self.read_extraction(image, box_overrides=box_overrides)
        return self._read_image(image, color_order, box_overrides, {"kind": "direct_aligned_image"})

    def _read_image(self, image, color_order, box_overrides, source):
        with self._lock:
            started = perf_counter()
            original = load_image(image, color_order=color_order)
            assessment = assess(original, self.config.fields, self.config.quality)
            images = {"original": original}
            operations = []
            if assessment["enhance"]:
                images["corrected"], operations = correct_card(original, assessment)
            result = OCRResult("needs_review", {}, quality={**assessment, "operations": operations},
                               source=source, images=images,
                               diagnostics={"ocr_executed": False, "config": asdict(self.config)})
            detections = {}
            for variant, card in list(images.items()):
                try:
                    boxes = self.detector.detect(card)
                    # Test/custom adapters must honor the same uniqueness contract.
                    if len({b.id for b in boxes}) != len(boxes):
                        raise InferenceError("Detector returned duplicate box IDs")
                    detections[variant] = boxes
                    result.detections[variant] = {"status": "ok", "boxes": [asdict(b) for b in boxes]}
                except ModelLoadError:
                    raise
                except Exception as exc:
                    detections[variant] = []
                    result.detections[variant] = {"status": "error", "error": str(exc), "boxes": []}
            selections = self._select_fields(original, detections, box_overrides or {})
            for spec in self.config.fields:
                selection = selections[spec.key]
                field_quality = assessment["fields"][spec.key]
                item = {"label": spec.label, "kind": spec.kind, "quality": field_quality,
                        "selection": {k: v for k, v in selection.items() if k != "boxes"},
                        "candidates": {}, "value": None, "status": "needs_review"}
                result.fields[spec.key] = item
                if selection["status"] != "ok":
                    item["reason"] = selection["reason"]
                    continue
                variants = ["original"]
                if field_quality["enhance"] and "corrected" in images:
                    variants.append("corrected")
                for variant in variants:
                    candidate = {"status": "error"}
                    item["candidates"][variant] = candidate
                    try:
                        native, zoomed, bounds, clipped = crop_field(
                            images[variant], selection["boxes"], pixel_zone(spec.zone, original.size),
                            self.config, manual=selection["manual"])
                        sharpened = (variant == "corrected" and self.config.quality.sharpen_amount > 0
                                     and ("soft_edges" in field_quality["reasons"]
                                          or self.config.quality.mode == "force"))
                        if sharpened:
                            zoomed = sharpen_crop(zoomed, self.config.quality.sharpen_amount)
                        image_key = f"{spec.key}/{variant}"
                        images[image_key] = zoomed
                        candidate.update(image_key=image_key, bounds=list(bounds),
                                         native_size=list(native.size), sharpened=sharpened,
                                         clipped_to_template=clipped)
                        raw = self.recognizer.recognize(zoomed)
                        if not isinstance(raw, str):
                            raise InferenceError("Recognizer must return a string")
                        result.diagnostics["ocr_executed"] = True
                        value, validation = interpret(raw, spec)
                        candidate.update(status="ok", text_raw=raw, value=value, validation=validation)
                    except ModelLoadError:
                        raise
                    except Exception as exc:
                        candidate.update(status="error", error=str(exc))
                item.update(decide(item["candidates"]))
            any_success = any(c["status"] == "ok" for f in result.fields.values()
                              for c in f["candidates"].values())
            if all(f["status"] in ("ok", "agreement") for f in result.fields.values()):
                result.status = "ok"
            elif not any_success and (any(d["status"] == "error" for d in result.detections.values())
                                      or any(f["candidates"] for f in result.fields.values())):
                result.status = "error"
            result.diagnostics.update(elapsed_seconds=perf_counter()-started,
                                      recognizer=getattr(self.recognizer, "info", {}))
            return result

    def _select_fields(self, original, detections, overrides):
        unknown = set(overrides) - {spec.key for spec in self.config.fields}
        if unknown:
            raise ValueError(f"Unknown field overrides: {sorted(unknown)}")
        selections, owners = {}, {}
        for spec in self.config.fields:
            override = overrides.get(spec.key)
            manual = override is not None
            source, explicit = "original", override
            if isinstance(override, Mapping):
                source, explicit = override.get("source", "original"), override.get("box_ids")
                if not isinstance(explicit, (list, tuple)):
                    raise ValueError("Manual override requires a box_ids list")
            selected, selected_source = [], None
            for variant in ([source] if manual else detections):
                if variant not in detections:
                    raise ValueError(f"No detection variant named {variant}")
                selected = select_boxes(detections[variant], pixel_zone(spec.zone, original.size),
                                        self.config.horizontal_overlap, explicit)
                if selected:
                    selected_source = variant
                    break
            selection = {"status": "ok" if selected else "no_box", "boxes": selected,
                         "box_ids": [b.id for b in selected], "geometry_source": selected_source,
                         "manual": manual, "reason": "" if selected else "No Paddle box selected for this field"}
            selections[spec.key] = selection
            for box in selected:
                owners.setdefault((selected_source, box.id), []).append(spec.key)
        for (source, bid), keys in owners.items():
            if len(keys) > 1:
                for key in keys:
                    selections[key].update(status="ambiguous_box",
                                           reason=f"{source} box {bid} selected for multiple fields: {keys}")
        return selections
