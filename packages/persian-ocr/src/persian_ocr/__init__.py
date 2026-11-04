"""Persian field OCR. Models and output files are initialized only on demand."""
from .config import FieldSpec, OCRConfig, QualityConfig, default_fields
from .errors import InferenceError, InvalidImageError, ModelLoadError, PersianOCRError
from .geometry import TextBox
from .pipeline import PersianOCR
from .result import OCRResult

__version__ = "0.1.1"
__all__ = ["PersianOCR", "OCRConfig", "QualityConfig", "FieldSpec", "default_fields",
           "OCRResult", "TextBox", "PersianOCRError", "InvalidImageError", "ModelLoadError",
           "InferenceError"]
