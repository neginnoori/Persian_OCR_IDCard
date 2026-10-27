import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from persian_card_ocr.pipeline import CardOCR, normalize_text, order_quad, resize_limit, validate_quad, warp_card


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.quad = np.float32([[20, 30], [280, 50], [260, 180], [30, 170]])

    def test_corner_order_and_invalid_geometry(self):
        np.testing.assert_allclose(order_quad(self.quad[[2, 0, 3, 1]]), self.quad)
        self.assertEqual(validate_quad(self.quad, (300, 200)), 'geometry_candidate')
        self.assertEqual(validate_quad(self.quad+500, (300, 200)), 'corners_outside_image')
        self.assertEqual(validate_quad(np.zeros((4, 2)), (300, 200)), 'invalid_quad')

    def test_warp_and_detector_size(self):
        image = Image.new('RGB', (300, 200), 'white')
        warp = warp_card(image, self.quad)
        self.assertGreater(warp.width, warp.height)
        self.assertEqual(resize_limit(Image.new('RGB', (4000, 1800)), 1280).size, (1280, 576))

    def test_normalization_and_reading_order(self):
        self.assertEqual(normalize_text('۱۲۳ ي ك'), '123 ی ک')
        regions = [{'box': [0, 0, 30, 20], 'text_raw': '۱۲۳'},
                   {'box': [50, 0, 100, 20], 'text_raw': 'شماره'}]
        self.assertEqual(CardOCR.reading_order(regions), 'شماره ۱۲۳')

    def test_uncertain_direction_and_failure_outputs(self):
        engine = CardOCR.__new__(CardOCR)
        engine.variant, engine.model_sha256 = 'mock', 'test'
        engine.locate = lambda image: (self.quad, [.9]*4)
        engine.read_regions = lambda image: [{'box': [10, 10, 80, 40], 'crop_box': [8, 8, 82, 42],
                                              'text': 'شاهین', 'text_raw': 'شاهین'}]
        with tempfile.TemporaryDirectory() as temp:
            result = engine.process(Image.new('RGB', (300, 200)), temp)
            self.assertEqual(result['status'], 'orientation_uncertain')
            self.assertEqual((Path(temp)/'text.txt').read_text(), 'شاهین')
            engine.locate = lambda image: (None, [.1])
            result = engine.process(Image.new('RGB', (300, 200)), temp)
            self.assertEqual(result['status'], 'corners_not_found')
            result = engine.process('/nonexistent/card.jpg', temp)
            self.assertEqual(result['status'], 'error')


if __name__ == '__main__':
    unittest.main()
