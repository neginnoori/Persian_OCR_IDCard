"""Card extraction only: no final field OCR, UI, plotting or automatic image writes."""
from threading import Lock
from .types import ExtractionResult, ExtractorConfig
from .errors import (
    CardIDError, InvalidImageError, GPURequiredError, ModelLoadError,
    InferenceError, CardNotFoundError, ReviewRequiredError,
)
from .extractor import CardExtractor

__version__ = "0.1.0"


_DEFAULT = None
_DEFAULT_LOCK = Lock()


def _default_extractor():
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = CardExtractor()
        return _DEFAULT


def extract_card(image, *, color_order="RGB"):
    """Return the extracted PIL RGB card, reusing one default GPU extractor per process.

    Raises CardNotFoundError/ReviewRequiredError instead of silently returning an uncertain crop.
    Use CardExtractor for custom settings or explicit startup/model lifetime control.
    """
    return _default_extractor().extract(image, color_order=color_order)


__all__ = ["CardExtractor", "ExtractorConfig", "ExtractionResult", "extract_card",
           "CardIDError", "InvalidImageError", "GPURequiredError", "ModelLoadError",
           "InferenceError", "CardNotFoundError", "ReviewRequiredError"]
