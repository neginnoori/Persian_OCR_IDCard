"""Image normalization without output writes or automatic geometry changes."""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .errors import InvalidImageError


def load_image(source, *, color_order: str = "RGB") -> Image.Image:
    if color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must be RGB or BGR")
    try:
        if isinstance(source, Image.Image):
            image = ImageOps.exif_transpose(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            if source.dtype != np.uint8 or source.ndim not in (2, 3):
                raise InvalidImageError("NumPy input must be uint8 grayscale or RGB/BGR")
            if source.ndim == 3 and source.shape[2] != 3:
                raise InvalidImageError("NumPy color input must have exactly 3 channels")
            array = source[..., ::-1] if source.ndim == 3 and color_order == "BGR" else source
            image = Image.fromarray(array).convert("RGB")
        else:
            if isinstance(source, (bytes, bytearray, memoryview)):
                source = BytesIO(bytes(source))
            elif isinstance(source, (str, Path)):
                source = Path(source).expanduser()
            elif not hasattr(source, "read"):
                raise InvalidImageError("Use a path, bytes, file object, PIL image or NumPy array")
            # Do not close a file-like object owned by the caller.
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        if min(image.size) < 32:
            raise InvalidImageError("Card image is too small (minimum dimension: 32 pixels)")
        return image.copy()
    except InvalidImageError:
        raise
    except Exception as exc:
        raise InvalidImageError(f"Could not read input image: {exc}") from exc
