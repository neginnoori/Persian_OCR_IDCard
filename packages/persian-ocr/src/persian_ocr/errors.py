"""Public errors: invalid input/configuration and runtime failures remain distinct."""


class PersianOCRError(Exception):
    """Base class for package errors."""


class InvalidImageError(PersianOCRError, ValueError):
    """Input cannot be decoded as an image."""


class ModelLoadError(PersianOCRError):
    """A model or its optional dependency could not be loaded."""


class InferenceError(PersianOCRError):
    """A model failed during inference."""
