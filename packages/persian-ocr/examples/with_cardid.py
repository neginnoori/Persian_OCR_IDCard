"""Install cardid-extractor separately; this extraction step needs CUDA."""
import sys
from cardid import CardExtractor, ExtractorConfig
from persian_ocr import PersianOCR

extractor = CardExtractor(ExtractorConfig(width=1712, height=1080))
engine = PersianOCR()
extraction = extractor.extract_result(sys.argv[1])
result = engine.read_extraction(extraction)
# A not_found result never loads Paddle/Hezar or enters the quality/OCR stages.
if result.status == "not_found":
    print("No card found; OCR skipped")
else:
    print(result.status, result.data)
    result.save_report("ocr-output/report")
