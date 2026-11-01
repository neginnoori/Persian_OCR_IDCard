from io import BytesIO
from os import PathLike
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .errors import InvalidImageError


def normalize_image(image, *, color_order="RGB") -> Image.Image:
    """Decode path/bytes/PIL/uint8 NumPy into independent, EXIF-corrected RGB pixels.

    Arrays accept HxW, HxWx1, HxWx3 and HxWx4. BGR/BGRA require color_order='BGR'.
    Single-channel values are copied into all three channels, never colorized.
    """
    if color_order not in ("RGB", "BGR"):
        raise InvalidImageError("color_order must be 'RGB' or 'BGR'")
    try:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                raise InvalidImageError("NumPy input must use uint8 pixels (0..255)")
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[:, :, 0]
            if image.ndim == 3 and image.shape[2] in (3, 4):
                if color_order == "BGR":
                    image = image[:, :, [2, 1, 0, 3]] if image.shape[2] == 4 else image[:, :, ::-1]
            elif image.ndim != 2:
                raise InvalidImageError("Expected HxW or HxWx1/3/4 NumPy image")
            if not image.size:
                raise InvalidImageError("Image is empty")
            pil = Image.fromarray(np.ascontiguousarray(image))
        elif isinstance(image, Image.Image):
            pil = image.copy()
        elif isinstance(image, (str, PathLike)):
            with Image.open(image) as opened:
                pil = opened.copy()
        elif isinstance(image, (bytes, bytearray, memoryview)):
            with Image.open(BytesIO(bytes(image))) as opened:
                pil = opened.copy()
        else:
            raise InvalidImageError("Expected an image path, bytes, PIL image, or uint8 NumPy array")
        if pil.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK"):
            raise InvalidImageError(f"Unsupported image mode {pil.mode}; convert high-depth images to uint8 first")
        pil = ImageOps.exif_transpose(pil)
        if min(pil.size) < 16:
            raise InvalidImageError("Image dimensions must both be >= 16 pixels")
        if "A" in pil.getbands() or (pil.mode == "P" and "transparency" in pil.info):
            rgba = pil.convert("RGBA")
            pil = Image.alpha_composite(Image.new("RGBA", rgba.size, (255,255,255,255)), rgba)
        return pil.convert("RGB")
    except InvalidImageError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise InvalidImageError(f"Cannot decode input image: {error}") from error
