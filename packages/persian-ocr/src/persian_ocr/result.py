"""Result objects contain data and in-memory images; exporting is always explicit."""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image

from .recognition import decide


@dataclass
class OCRResult:
    status: str
    fields: dict
    quality: dict = field(default_factory=dict)
    detections: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    images: dict[str, Image.Image] = field(default_factory=dict, repr=False)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def data(self) -> dict[str, str | None]:
        return {key: value.get("value") for key, value in self.fields.items()}

    @property
    def ok(self):
        return self.status == "ok"

    def to_dict(self, *, detailed: bool = False):
        if not detailed:
            return self.data
        return deepcopy({"schema_version": "1.0", "status": self.status, "created_at": self.created_at,
                         "data": self.data, "fields": self.fields, "quality": self.quality,
                         "detections": self.detections, "diagnostics": self.diagnostics,
                         "source": self.source})

    def select_variant(self, field_key: str, variant: str):
        """Choose an observed reading explicitly, without running OCR again."""
        item = self.fields[field_key]
        item.update(decide(item.get("candidates", {}), variant))
        self.status = "ok" if all(f["status"] in ("ok", "agreement", "user_selected")
                                  for f in self.fields.values()) else "needs_review"
        if self.diagnostics.get("extraction_needs_review"):
            self.status = "needs_review"
        return self

    def save_json(self, path, *, detailed: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(detailed=detailed), ensure_ascii=False,
                                   indent=2, allow_nan=False), encoding="utf-8")
        return path

    def save_report(self, directory) -> dict[str, Path]:
        from .report import save_report
        return save_report(self, Path(directory))
