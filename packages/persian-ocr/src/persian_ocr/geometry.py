"""Field assignment and pixel-preserving native crops."""
from dataclasses import dataclass
import math

from PIL import Image


@dataclass(frozen=True)
class TextBox:
    id: int
    polygon: tuple[tuple[float, float], ...]
    score: float

    def __post_init__(self):
        if len(self.polygon) != 4 or any(
            len(p) != 2 or not all(math.isfinite(v) for v in p) for p in self.polygon
        ):
            raise ValueError("A text box requires four finite (x, y) points")
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Detection score must be finite and in [0, 1]")
        x1, y1, x2, y2 = self.bounds
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Empty text box")

    @property
    def bounds(self):
        return (min(p[0] for p in self.polygon), min(p[1] for p in self.polygon),
                max(p[0] for p in self.polygon), max(p[1] for p in self.polygon))


def pixel_zone(zone, size):
    return (zone[0] * size[0], zone[1] * size[1], zone[2] * size[0], zone[3] * size[1])


def select_boxes(boxes, zone, overlap, explicit=None):
    if explicit is not None:
        if any(type(i) is not int for i in explicit) or len(set(explicit)) != len(explicit):
            raise ValueError("Box IDs must be unique integers")
        lookup = {b.id: b for b in boxes}
        if set(explicit) - lookup.keys():
            raise ValueError("Unknown box ID in manual selection")
        return [lookup[i] for i in explicit]
    selected = []
    for box in boxes:
        x1, y1, x2, y2 = box.bounds
        width = max(0, min(x2, zone[2]) - max(x1, zone[0]))
        if zone[1] <= (y1+y2)/2 <= zone[3] and width / (x2-x1) >= overlap:
            selected.append(box)
    return selected


def crop_field(image, boxes, zone, config, *, manual=False):
    bb = [box.bounds for box in boxes]
    if len(bb) > 1 and min(b[3] for b in bb) <= max(b[1] for b in bb):
        raise ValueError("Selected field boxes span multiple rows")
    union = (min(b[0] for b in bb), min(b[1] for b in bb),
             max(b[2] for b in bb), max(b[3] for b in bb))
    limit = (0, 0, image.width, image.height) if manual else zone
    clipped = (max(limit[0], union[0]), max(limit[1], union[1]),
               min(limit[2], union[2]), min(limit[3], union[3]))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("Field has no intersection with the card window")
    pad = max(2, round((clipped[3]-clipped[1]) * config.box_padding))
    bounds = (max(0, math.floor(clipped[0]-pad)), max(0, math.floor(clipped[1]-pad)),
              min(image.width, math.ceil(clipped[2]+pad)),
              min(image.height, math.ceil(clipped[3]+pad)))
    native = image.crop(bounds)
    scale = min(config.crop_zoom, 2048 / max(native.size))
    zoomed = native.resize(tuple(max(2, round(v*scale)) for v in native.size),
                           Image.Resampling.LANCZOS)
    return native, zoomed, bounds, any(abs(a-b) > 1 for a, b in zip(union, clipped))
