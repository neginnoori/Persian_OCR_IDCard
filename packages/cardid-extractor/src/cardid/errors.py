"""Explicit errors for callers; model/runtime failures are never disguised as no detection."""


class CardIDError(RuntimeError):
    """Base error for this package."""


class InvalidImageError(CardIDError):
    """Input cannot be decoded or is not a supported image."""


class GPURequiredError(CardIDError):
    """CUDA is absent, unusable, or a model is on the wrong device."""


class ModelLoadError(CardIDError):
    """Weights or inference dependencies could not be loaded."""


class InferenceError(CardIDError):
    """A model failed during inference."""


class CardNotFoundError(CardIDError):
    """No card detected in the supplied image."""


class ReviewRequiredError(CardIDError):
    """A candidate exists, but geometry or orientation needs review."""

    def __init__(self, result):
        self.result = result
        super().__init__(result.reason or "Card geometry or orientation requires review")
