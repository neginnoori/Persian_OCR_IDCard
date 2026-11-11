from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from PIL import Image


@dataclass(frozen=True)
class ExtractorConfig:
    """Inference settings. GPU 0 is required; no implicit CPU fallback."""

    device: str = "cuda"
    cache_dir: str | Path | None = None
    offline: bool = False
    width: int = 856
    height: int = 540
    box_threshold: float = .25
    text_threshold: float = .20
    min_quad_iou: float = .88
    min_orientation_margin: float = 1.2
    candidate_selection: str = "legacy"

    def __post_init__(self):
        if self.candidate_selection not in ("legacy", "v2", "v3"):
            raise ValueError("candidate_selection must be legacy, v2 or v3")
        if self.device not in ("cuda", "cuda:0"):
            raise ValueError("Only CUDA GPU 0 is supported: device='cuda' or 'cuda:0'")
        if not isinstance(self.width, int) or not isinstance(self.height, int) or min(self.width, self.height) < 32:
            raise ValueError("width and height must be integers >= 32")
        for name in ("box_threshold", "text_threshold", "min_quad_iou"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.min_orientation_margin < float('inf'):
            raise ValueError("min_orientation_margin must be finite and >= 0")


@dataclass
class ExtractionResult:
    """image is always a PIL RGB image, or None when no crop can be produced.

    A needs_review image is only a candidate. Do not send it to OCR automatically.
    Corner/H coordinates refer to the EXIF-corrected input, not undecoded file pixels.
    """

    status: Literal["ok", "needs_review", "not_found"]
    image: Image.Image | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe diagnostics, deliberately excluding image pixels."""
        return {"status": self.status, "reason": self.reason, "metadata": self.metadata}

