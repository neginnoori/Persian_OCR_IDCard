"""Run in the installed Python environment: python examples/basic.py card.png."""
import sys
from persian_ocr import PersianOCR

engine = PersianOCR()  # CPU, lazy model loading; reuse for subsequent cards
result = engine.read(sys.argv[1])  # aligned card; accepts image bytes or PIL too
data = result.data
print(result.status, data)
result.save_json("ocr-output/data.json")
paths = result.save_report("ocr-output/report")
print(paths["html"])
