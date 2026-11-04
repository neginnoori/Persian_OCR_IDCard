"""Contract and pipeline tests with deterministic models; no weights/network required."""
from contextlib import nullcontext
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from persian_ocr import (FieldSpec, InvalidImageError, ModelLoadError, OCRConfig,
                         PersianOCR, QualityConfig, TextBox)
from persian_ocr.backends import HezarRecognizer, PaddleDetector
from persian_ocr.cli import main as cli_main
from persian_ocr.images import load_image
from persian_ocr.recognition import interpret

VALUES = ["مینا", "آزمایشی", "بهرام", "۰۰۵۶۳۴۰۱۲۳", "۱۳۷۵/۰۸/۰۳", "۱۴۰۵/۰۸/۱۹"]
COORDS = [(700,260,900,284), (700,330,900,352), (700,465,900,488),
          (680,180,900,212), (690,398,900,420), (685,535,900,558)]


def boxes():
    return [TextBox(i, ((x1,y1),(x2,y1),(x2,y2),(x1,y2)), .9)
            for i, (x1,y1,x2,y2) in enumerate(COORDS,1)]


def card(dark=False):
    array = np.full((720,1140,3),190,dtype=np.uint8)
    for x,y,_,_ in COORDS:
        cv2.putText(array,"SAMPLE123",(x,y+24),cv2.FONT_HERSHEY_SIMPLEX,.85,(25,25,25),2,cv2.LINE_AA)
    if dark:
        array = (array.astype(float)*.4).astype(np.uint8)
    return Image.fromarray(array)


class Detector:
    def __init__(self, empty_first=False):
        self.calls = 0
        self.empty_first = empty_first

    def detect(self, image):
        self.calls += 1
        return [] if self.empty_first and self.calls == 1 else boxes()


class Recognizer:
    def __init__(self, values=None):
        self.calls = 0
        self.values = VALUES if values is None else values

    def recognize(self, image):
        assert isinstance(image, Image.Image)
        value = self.values[self.calls % len(self.values)]
        self.calls += 1
        return value


class PackageTests(unittest.TestCase):
    def engine(self, config=None, values=None, detector=None):
        det, rec = detector or Detector(), Recognizer(values)
        return PersianOCR(config, detector=det, recognizer=rec), det, rec

    def test_good_card_original_only_and_input_unchanged(self):
        engine, det, rec = self.engine()
        image = card()
        before = image.tobytes()
        result = engine.read(image)
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(det.calls,1)
        self.assertEqual(rec.calls,6)
        self.assertNotIn("corrected", result.images)
        self.assertEqual(result.status,"ok")
        self.assertEqual(result.data["national_id"],"0056340123")
        self.assertEqual(result.data,result.to_dict())

    def test_dark_card_two_versions_same_geometry_and_conflict(self):
        values = [v for value in VALUES for v in (value,value)]
        values[7] = "۰۰۵۶۳۴۰۱۲۴"
        engine, det, rec = self.engine(values=values)
        result = engine.read(card(dark=True))
        self.assertEqual((det.calls,rec.calls),(2,12))
        self.assertTrue(result.quality["enhance"])
        self.assertIsNone(result.data["national_id"])
        self.assertEqual(result.fields["national_id"]["status"],"conflict")
        candidates = result.fields["national_id"]["candidates"]
        self.assertEqual(candidates["original"]["bounds"],candidates["corrected"]["bounds"])
        self.assertGreater(np.median(np.array(result.images["corrected"])),np.median(np.array(result.images["original"])))
        result.select_variant("national_id","corrected")
        self.assertEqual(result.data["national_id"],"0056340124")
        self.assertEqual(rec.calls,12)

    def test_off_mode_keeps_dark_image_original_only(self):
        engine, det, rec = self.engine(OCRConfig(quality=QualityConfig(mode="off")))
        result = engine.read(card(dark=True))
        self.assertFalse(result.quality["enhance"])
        self.assertEqual((det.calls,rec.calls),(1,6))

    def test_dark_portrait_does_not_trigger_field_quality(self):
        image = np.array(card())
        image[:,:500] = 0
        engine, _, _ = self.engine()
        self.assertFalse(engine.read(image).quality["enhance"])

    def test_memory_only_result_creates_no_output_files(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = Path.cwd()
            try:
                os.chdir(temp)
                engine, _, _ = self.engine()
                engine.read(card())
                self.assertEqual(list(Path(temp).iterdir()),[])
            finally:
                os.chdir(previous)

    def test_not_found_none_and_unknown_status_never_enter_ocr(self):
        engine, det, rec = self.engine()
        for source in (None, {"status":"not_found"}, {"status":"no card found"},
                       {"status":"no photos found", "image":card()},
                       {"status":"unexpected", "image":card()},
                       types.SimpleNamespace(status="not_found",image=card(),reason="none")):
            with patch("persian_ocr.pipeline.assess") as quality:
                result = engine.read(source)
            quality.assert_not_called()
            self.assertFalse(result.diagnostics["ocr_executed"])
            self.assertTrue(all(v is None for v in result.data.values()))
        self.assertEqual((det.calls,rec.calls),(0,0))

    def test_needs_review_runs_by_default_and_strict_opt_out_remains_available(self):
        engine, det, _ = self.engine()
        extraction = types.SimpleNamespace(status="needs_review",image=card(),reason="orientation")
        self.assertEqual(engine.read_extraction(extraction,approve_review=False).status,"needs_review")
        self.assertEqual(det.calls,0)
        result = engine.read_extraction(extraction)
        self.assertEqual(det.calls,1)
        self.assertEqual(result.status,"needs_review")
        self.assertTrue(result.diagnostics["ocr_executed"])
        self.assertTrue(result.diagnostics["extraction_needs_review"])
        self.assertEqual(result.diagnostics["extraction_reason"],"orientation")
        self.assertEqual(result.data["first_name"],VALUES[0])
        result.select_variant("first_name","original")
        self.assertEqual(result.status,"needs_review")
        self.assertEqual(engine.read_extraction({"status":"not_found"},approve_review=True).status,"not_found")

    def test_review_without_an_image_skips_models_with_explicit_reason(self):
        engine, det, rec = self.engine()
        with patch("persian_ocr.pipeline.assess") as quality:
            result = engine.read_extraction({"status":"needs_review", "reason":"no_reliable_quadrilateral"})
        quality.assert_not_called()
        self.assertEqual((det.calls,rec.calls),(0,0))
        self.assertEqual(result.diagnostics["reason"],"extraction_image_missing")

    def test_read_and_raw_photo_both_process_review_images(self):
        engine, _, rec = self.engine()
        extraction = {"status":"needs_review", "image":card(), "reason":"orientation"}
        self.assertTrue(engine.read(extraction).diagnostics["ocr_executed"])
        engine.extractor = types.SimpleNamespace(extract_result=lambda image: extraction)
        self.assertTrue(engine.read_photo(card()).diagnostics["ocr_executed"])
        self.assertEqual(rec.calls,12)

    def test_pixel_free_success_dict_is_rejected(self):
        engine, det, _ = self.engine()
        with self.assertRaises(InvalidImageError):
            engine.read_extraction({"status":"ok","metadata":{}})
        self.assertEqual(det.calls,0)

    def test_raw_photo_integration_gates_extractor_result(self):
        extractor = types.SimpleNamespace(extract_result=lambda image: {"status":"not_found"})
        engine, det, rec = self.engine()
        engine.extractor = extractor
        result = engine.read_photo(card())
        self.assertEqual(result.status,"not_found")
        self.assertEqual((det.calls,rec.calls),(0,0))
        extractor.extract_result = lambda image: {"status":"ok","image":image}
        self.assertEqual(engine.read_photo(card()).status,"ok")

    def test_bytes_filelike_path_and_bgr_inputs(self):
        image = card()
        buffer = BytesIO()
        image.save(buffer,format="PNG")
        for source in (buffer.getvalue(),BytesIO(buffer.getvalue()),np.array(image)):
            np.testing.assert_array_equal(np.array(load_image(source)),np.array(image))
        np.testing.assert_array_equal(np.array(load_image(np.array(image)[...,::-1],color_order="BGR")),np.array(image))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"کارت نمونه.png"
            image.save(path)
            self.assertEqual(load_image(path).size,image.size)
        with self.assertRaises(InvalidImageError):
            load_image(b"not an image")
        with self.assertRaises(InvalidImageError):
            load_image(np.zeros((40,40,3),dtype=float))

    def test_blank_detection_returns_missing_fields_without_recognition(self):
        engine, _, rec = self.engine()
        engine.detector.detect = lambda image: []
        result = engine.read(card())
        self.assertEqual(rec.calls,0)
        self.assertEqual(result.status,"needs_review")
        self.assertTrue(all(v is None for v in result.data.values()))

    def test_corrected_detection_fallback_uses_same_geometry(self):
        engine, _, _ = self.engine(detector=Detector(empty_first=True),
                                   values=[v for value in VALUES for v in (value,value)])
        result = engine.read(card(dark=True))
        self.assertTrue(all(f["selection"]["geometry_source"]=="corrected" for f in result.fields.values()))

    def test_shared_box_rejected_and_manual_unknown_ids_visible(self):
        engine, _, rec = self.engine()
        result = engine.read(card(),box_overrides={"first_name":[1],"last_name":[1]})
        self.assertEqual(result.fields["first_name"]["selection"]["status"],"ambiguous_box")
        self.assertEqual(result.fields["last_name"]["selection"]["status"],"ambiguous_box")
        self.assertEqual(rec.calls,4)
        with self.assertRaises(ValueError):
            engine.read(card(),box_overrides={"first_name":[999]})

    def test_invalid_config_and_detection_geometry_rejected(self):
        for factory in (lambda: QualityConfig(mode="wrong"),lambda: QualityConfig(dark_median=float("nan")),
                        lambda: FieldSpec("x","x",(.8,0,.2,1)),lambda: OCRConfig(crop_zoom=0),
                        lambda: TextBox(1,((0,0),(0,0),(0,0),(0,0)),.9)):
            with self.assertRaises(ValueError):
                factory()

    def test_model_load_errors_are_not_hidden_as_bad_quality(self):
        engine, _, _ = self.engine()
        def fail(image):
            raise ModelLoadError("test failure")
        engine.recognizer.recognize = fail
        with self.assertRaises(ModelLoadError):
            engine.read(card())

    def test_json_and_portable_report_with_status_colored_boxes(self):
        values = ["<script>"]+VALUES[1:]
        engine, _, _ = self.engine(values=values)
        result = engine.read(card())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = result.save_json(root/"داده.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),result.data)
            self.assertNotIn("image",path.read_text(encoding="utf-8"))
            files = result.save_report(root/"گزارش")
            html = files["html"].read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;",html)
            self.assertNotIn("<script>",html)
            self.assertIn("data:image/png;base64,",html)
            with Image.open(files["original_image"]) as marked:
                self.assertEqual(marked.getpixel((700,270)),(15,160,60))
            details = json.loads(files["details"].read_text(encoding="utf-8"))
            self.assertEqual(details["data"],result.data)
            self.assertEqual(details["fields"]["national_id"]["selection"]["box_ids"],[4])

    def test_report_for_skipped_result_has_no_fabricated_image(self):
        engine, _, _ = self.engine()
        with tempfile.TemporaryDirectory() as temp:
            paths = engine.read(None).save_report(temp)
            self.assertNotIn("original_image",paths)
            self.assertTrue(paths["html"].is_file())

    def test_cli_writes_dictionary_or_details_and_errors_are_json(self):
        engine, _, _ = self.engine()
        result = engine.read(card())
        with patch("persian_ocr.cli.PersianOCR") as klass, patch("sys.stdout",new_callable=StringIO) as stdout:
            klass.return_value.read.return_value = result
            exit_code = cli_main(["input.png","--detailed"])
        self.assertEqual(exit_code,0)
        self.assertEqual(json.loads(stdout.getvalue())["data"],result.data)
        with patch("sys.stdout",new_callable=StringIO) as stdout:
            exit_code = cli_main(["a_file_that_does_not_exist.png"])
        self.assertEqual(exit_code,2)
        self.assertEqual(json.loads(stdout.getvalue())["status"],"error")

    def test_digit_normalization_preserves_order_and_leading_zeros(self):
        config = OCRConfig()
        spec = next(f for f in config.fields if f.key=="national_id")
        value, validation = interpret("\u200f۰۰۵۶۳۴ ٠١٢٣",spec)
        self.assertEqual(value,"0056340123")
        self.assertTrue(validation["format_ok"])
        date = next(f for f in config.fields if f.kind=="date")
        self.assertFalse(interpret("1375/13/03",date)[1]["format_ok"])

    def test_import_does_not_import_torch_paddle_or_create_files(self):
        code = "import sys,persian_ocr; assert 'torch' not in sys.modules; assert 'paddleocr' not in sys.modules; print('lazy import OK')"
        process = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)


class AdapterTests(unittest.TestCase):
    def test_paddle_contract_bgr_and_single_model_load(self):
        calls = []
        class Native:
            def __init__(self,**kwargs):
                calls.append(kwargs)
            def predict(self,input,batch_size):
                calls.append(input[0,0].tolist())
                yield types.SimpleNamespace(json={"res":{"dt_polys":[[[1,1],[20,1],[20,10],[1,10]]],"dt_scores":[.9]}})
        backend = PaddleDetector(OCRConfig())
        with patch.dict(sys.modules,{"paddleocr":types.SimpleNamespace(TextDetection=Native)}):
            result = backend.detect(Image.new("RGB",(40,40),(1,2,3)))
            backend.detect(Image.new("RGB",(40,40),(1,2,3)))
        self.assertEqual(len([v for v in calls if isinstance(v,dict)]),1)
        self.assertEqual(calls[1],[3,2,1])
        self.assertEqual(result[0].id,1)

    def test_hezar_pil_input_and_snapshot_without_global_dtype_changes(self):
        seen = []
        class Model:
            config = types.SimpleNamespace(reverse_output_digits=True)
            @classmethod
            def load(cls,*args,**kwargs):
                seen.append((args,kwargs))
                return cls()
            def eval(self):
                return self
            def predict(self,images,**kwargs):
                self.assert_image = isinstance(images[0],Image.Image)
                return [types.SimpleNamespace(to_dict=lambda: {"text":"۰۰۵۶۳۴۰۱۲۳"})]
        torch = types.SimpleNamespace(inference_mode=nullcontext)
        fake_modules = {"torch":torch,"hezar":types.ModuleType("hezar"),"hezar.models":types.SimpleNamespace(Model=Model),
                        "huggingface_hub":types.SimpleNamespace(snapshot_download=lambda *a,**kw:"/cache/snapshots/abc123")}
        backend = HezarRecognizer(OCRConfig())
        with patch.dict(sys.modules,fake_modules):
            self.assertEqual(backend.recognize(card()),"۰۰۵۶۳۴۰۱۲۳")
            backend.recognize(card())
        self.assertEqual(len(seen),1)
        self.assertEqual(backend.info["resolved_snapshot"],"abc123")
        self.assertTrue(backend._model.assert_image)


if __name__ == "__main__":
    unittest.main(verbosity=2)
