from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from cardid.images import normalize_image
from cardid import InvalidImageError


@pytest.mark.parametrize("shape",[(40,50),(40,50,1),(40,50,3),(40,50,4)])
def test_grayscale_array_normalization(shape):
    array = np.full(shape,127,np.uint8)
    if len(shape)==3 and shape[2]==4: array[:,:,3] = 255
    out = normalize_image(array)
    assert out.mode=='RGB' and out.size==(50,40)
    assert np.array_equal(np.asarray(out),np.full((40,50,3),127,np.uint8))


def test_bgr_and_alpha():
    out = normalize_image(np.tile(np.array([10,20,30],np.uint8),(30,40,1)),color_order='BGR')
    assert out.getpixel((0,0))==(30,20,10)
    rgba = np.zeros((30,40,4),np.uint8)
    assert normalize_image(rgba).getpixel((0,0))==(255,255,255)


def test_bytes_path_and_pil_are_equivalent(tmp_path):
    original = Image.new('L',(50,40),85)
    stream = BytesIO(); original.save(stream,format='PNG')
    path = tmp_path/'gray.png'; path.write_bytes(stream.getvalue())
    variants = [normalize_image(original),normalize_image(path),normalize_image(stream.getvalue())]
    assert all(v.mode=='RGB' and v.getpixel((0,0))==(85,85,85) for v in variants)
    assert original.mode=='L'


def test_exif_orientation_is_applied_before_inference(tmp_path):
    image = Image.new('RGB',(60,30),'red')
    exif=image.getexif(); exif[274]=6
    path=tmp_path/'rotated.jpg'; image.save(path,exif=exif)
    assert normalize_image(path).size==(30,60)
    with Image.open(path) as source:
        assert normalize_image(source).size==(30,60)


@pytest.mark.parametrize('bad',[
    b'not a photo',None,np.zeros((20,20),np.float32),np.zeros((20,20,2),np.uint8),
    np.zeros((0,20),np.uint8),np.zeros((2,2),np.uint8),
])
def test_invalid_input_rejected(bad):
    with pytest.raises(InvalidImageError): normalize_image(bad)
