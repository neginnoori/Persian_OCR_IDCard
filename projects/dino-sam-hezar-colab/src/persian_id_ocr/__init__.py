"""Reusable extraction and Persian OCR; model loading is lazy."""
from cardid import CardExtractor, ExtractorConfig, extract_card
from .pipeline import PersianCardOCR, OCRResult
from .reader import HezarReader

__all__ = ["CardExtractor", "ExtractorConfig", "extract_card", "PersianCardOCR", "OCRResult", "HezarReader"]
