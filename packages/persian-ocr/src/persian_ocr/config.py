"""Validated configuration and normalized field templates."""
from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    zone: tuple[float, float, float, float]
    kind: str = "text"

    def __post_init__(self):
        object.__setattr__(self, "zone", tuple(self.zone))
        if not self.key or not self.label or self.kind not in ("text", "national_id", "date"):
            raise ValueError("Invalid field key, label or kind")
        if len(self.zone) != 4 or not all(isfinite(x) for x in self.zone):
            raise ValueError("Field zone requires four finite numbers")
        left, top, right, bottom = self.zone
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("Field zones must be ordered normalized coordinates in [0, 1]")


def default_fields() -> tuple[FieldSpec, ...]:
    """Front layout used in the v13 notebook; override for other card layouts."""
    return (
        FieldSpec("first_name", "نام", (.47, .32, .83, .42)),
        FieldSpec("last_name", "نام خانوادگی", (.47, .42, .83, .52)),
        FieldSpec("father_name", "نام پدر", (.47, .62, .83, .72)),
        FieldSpec("national_id", "کد ملی", (.47, .21, .83, .34), "national_id"),
        FieldSpec("birth_date", "تاریخ تولد", (.47, .49, .83, .63), "date"),
        FieldSpec("expiry_date", "پایان اعتبار", (.47, .68, .83, .82), "date"),
    )


@dataclass(frozen=True)
class QualityConfig:
    mode: str = "auto"
    dark_median: float = 105.
    low_contrast: float = 40.
    blur_laplacian: float = 6.
    uneven_light: float = 50.
    clahe_clip: float = 1.5
    clahe_blend: float = .5
    sharpen_amount: float = .25

    def __post_init__(self):
        if self.mode not in ("auto", "off", "force"):
            raise ValueError("Quality mode must be auto, off or force")
        for name, low, high in (
            ("dark_median", 0, 255), ("low_contrast", 0, 255),
            ("blur_laplacian", 0, 1e6), ("uneven_light", 0, 255),
            ("clahe_clip", .1, 4), ("clahe_blend", 0, 1), ("sharpen_amount", 0, .5),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid quality setting: {name}")


@dataclass(frozen=True)
class OCRConfig:
    fields: tuple[FieldSpec, ...] = field(default_factory=default_fields)
    quality: QualityConfig = field(default_factory=QualityConfig)
    crop_zoom: float = 2.
    box_padding: float = .12
    horizontal_overlap: float = .35
    detection_min_score: float = 0.
    detector_device: str = "cpu"
    recognizer_device: str = "cpu"
    detector_model: str = "PP-OCRv5_server_det"
    detector_model_dir: str | None = None
    recognizer_model: str = "hezarai/crnn-base-fa-v2"
    recognizer_revision: str = "main"
    cache_dir: str | None = None
    offline: bool = False

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.fields or len({f.key for f in self.fields}) != len(self.fields):
            raise ValueError("Field keys must be nonempty and unique")
        for name, low, high in (
            ("crop_zoom", 1, 4), ("box_padding", 0, .4),
            ("horizontal_overlap", .01, 1), ("detection_min_score", 0, 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid OCR setting: {name}")
        if self.recognizer_device not in ("cpu", "cuda"):
            raise ValueError("Recognizer device must be cpu or cuda")
