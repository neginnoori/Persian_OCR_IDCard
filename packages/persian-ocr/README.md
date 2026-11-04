# Persian Card OCR

پکیج OCR فیلدهای روی کارت ملی ایران، بر پایهٔ مسیر آزموده‌شده در نوت‌بوک ۱۳:

**کارت آماده → بررسی شرطی کیفیت → محدودهٔ فیلدها → Paddle TextDetection → برش → Hezar → داده/گزارش**

`Paddle` فقط محل متن را پیدا می‌کند؛ خواندن نام‌ها و اعداد با `hezarai/crnn-base-fa-v2` است. کلاس اصلی `PersianOCR` است. پکیج به Colab وابسته نیست و فایل نتیجه را فقط با درخواست صریح ذخیره می‌کند. مدل‌ها در اولین استفاده بارگیری و در همان نمونهٔ موتور دوباره استفاده می‌شوند.

## نصب در VS Code و سیستم محلی

پروفایل فعلی برای **Python 3.11 یا 3.12، Windows/Linux با معماری x86-64** بسته شده است. برای شروع Python 3.12 و یک محیط مجازی تازه پیشنهاد می‌شود. در VS Code همان محیط را با Select Interpreter انتخاب کنید.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install ".[models]"
.venv\Scripts\python -m pip check
```

در Linux، ساخت محیط با `python3.12 -m venv .venv` و مسیر Python با `.venv/bin/python` است. برای CUDA، PyTorch مناسب GPU را پیش از نصب `[models]` نصب کنید؛ مسیر CPU برای خود OCR کافی است. اولین اجرای واقعی وزن مدل‌ها را دانلود می‌کند. وجود VS Code به‌تنهایی به معنی نصب وابستگی‌ها نیست.

اگر Python سازگار ندارید، یا kernel فعلی Colab نسخهٔ دیگری است، اسکریپت زیر یک Python 3.12 جدا نصب می‌کند و kernel فعلی را تغییر نمی‌دهد:

```text
python scripts/setup_runtime.py --root .ocr-runtime
```

آخرین خط خروجی، JSON مسیر Python محیط است. در Windows مسیر معمول آن `.ocr-runtime/env/Scripts/python.exe` و در Linux `.ocr-runtime/env/bin/python` است. برای استخراج خام با CardID و GPU، گزینهٔ `--torch-index cu126` وجود دارد. این setup نیاز به اینترنت دارد؛ خودش مدل‌ها را برای OCR اجرا نمی‌کند. نصب سادهٔ خود کتابخانه بدون مدل‌ها: `pip install .`؛ برای اجرای واقعی backendها `[models]` لازم است.

## استفاده و سه نوع خروجی

ورودی مستقیم `read` باید **تصویر خود کارتِ استخراج‌شده، صاف و با جهت صحیح** باشد. برای عکس صحنه/دست/میز از `read_photo` یا CardID استفاده کنید؛ پنجره‌های فیلد روی کل کارت تعریف شده‌اند.

```python
from persian_ocr import PersianOCR

ocr = PersianOCR()  # CPU by default
result = ocr.read("aligned_card.jpg")

# ۳. داده در حافظه، قابل استفاده در کد؛ هیچ فایل نتیجه‌ای ساخته نشده است
data = result.data
first_name = data["first_name"]

# ۱. JSON سادهٔ داده، بدون عکس و جزئیات پردازش
result.save_json("outputs/data.json")

# ۲. گزارش همراه تصویر باکس‌ها و وضعیت هر فیلد
paths = result.save_report("outputs/report")
print(paths["html"])
```

`data.json` و `result.data` شکل یکسان دارند؛ مقادیر ناموجود/مبهم `None` در Python و `null` در JSON هستند. کد ملی همیشه رشته است و صفر آغازین حفظ می‌شود:

```json
{
  "first_name": null,
  "last_name": null,
  "father_name": null,
  "national_id": null,
  "birth_date": null,
  "expiry_date": null
}
```

این مثال ساختار است، نه خروجی واقعی OCR. `result.to_dict()` هم دیکشنری ساده را می‌دهد. `result.to_dict(detailed=True)` یا `save_json(..., detailed=True)` وضعیت، علت کیفیت، باکس‌های واقعی، خام‌خوانش‌ها، هندسهٔ برش و منبع ورودی را هم برمی‌گرداند؛ دادهٔ پیکسلی داخل JSON قرار نمی‌گیرد.

گزارش شامل `report.html` (تصاویر داخل HTML جاسازی شده‌اند)، `original_boxes.png`، در صورت اصلاح `corrected_boxes.png`، `data.json` و `details.json` است. باکس سبز: انتخاب‌شده و دارای خوانش پذیرفته‌شده؛ نارنجی: انتخاب‌شده و نیازمند بررسی؛ خاکستری: استفاده‌نشده. سبز بودن، تایید هویت یا اثبات صحت OCR نیست.

## آپلود، bytes، PIL و OpenCV

```python
# bytes حاصل از آپلود یک وب‌اپ یا Google Colab
result = ocr.read(uploaded_bytes)

# PIL.Image و file-like هم قابل استفاده‌اند
result = ocr.read(pil_image)

# آرایهٔ OpenCV به صورت پیش‌فرض BGR است
result = ocr.read(cv2_image, color_order="BGR")
```

PIL/bytes/file path با جهت EXIF و RGB یکسان می‌شوند. NumPy پیش‌فرض RGB است و باید uint8 باشد. تصویر ورودی تغییر نمی‌کند. نتایج تصویری در `result.images` نگهداری می‌شوند؛ برای آزاد شدن حافظه پس از مصرف، مرجع نتیجه را آزاد کنید.

## اتصال به cardid-extractor و توقف no-card

CardID اختیاری و مستقل است؛ آن را از ZIP پکیج خودش نصب کنید. استخراج آن CUDA می‌خواهد، حتی اگر OCR روی CPU اجرا شود.

```python
from cardid import CardExtractor, ExtractorConfig
from persian_ocr import PersianOCR

extractor = CardExtractor(ExtractorConfig(width=1712, height=1080))
ocr = PersianOCR()

card = extractor.extract_result("camera_photo.jpg")
result = ocr.read_extraction(card)  # همچنین ocr.read(card)
```

اگر `status="not_found"` یا `"no_card_found"` / `"no_photos_found"` باشد (یا نتیجه None باشد)، بررسی کیفیت و OCR اصلاً اجرا نمی‌شوند و مدل‌های OCR بارگذاری نمی‌شوند؛ `result.status == "not_found"` و داده‌ها خالی/null هستند. `needs_review` از نسخهٔ ۰.۱.۱، اگر تصویر کارت داشته باشد، پیش‌فرض وارد OCR می‌شود؛ داده خوانده می‌شود اما وضعیت نیازمند بررسی و دلیل استخراج در نتیجه حفظ می‌شود. اگر تصویر همراه این وضعیت نباشد، OCR ممکن نیست و دلیل `extraction_image_missing` ثبت می‌شود. برای بازگرداندن رفتار سخت‌گیرانهٔ قبلی:

```python
result = ocr.read_extraction(card, approve_review=False)  # skip needs_review
```

حتی با `approve_review=True` نتیجهٔ `not_found` وارد OCR نمی‌شود. به API خود `ExtractionResult` را بدهید؛ `card.to_dict()` عکس را ندارد و برای ورود نتیجهٔ موفق کافی نیست. ارسال مستقیم `card.image` وقتی مناسب است که تصویر کارت واقعاً موجود باشد؛ استفاده از `read_extraction` وضعیت اولیه را هم در گزارش نگه می‌دارد.

روش کوتاه برای عکس خام:

```python
ocr = PersianOCR(extractor=extractor)
result = ocr.read_photo("camera_photo.jpg")
```

بدون extractor تزریق‌شده، `read_photo` به صورت lazy یک `cardid.CardExtractor` می‌سازد. استخراج‌کنندهٔ سفارشی فقط باید `extract_result(PIL_image)` داشته باشد و نتیجهٔ status/image بدهد.

## اصلاح شرطی کیفیت و اختلاف خوانش‌ها

منطق و پیش‌فرض‌ها از نسخهٔ ۱۳ حفظ شده‌اند. معیارها روی پنجرهٔ فیلدها هستند، نه عکس چهره یا زمینهٔ اطراف کارت:

* میانهٔ روشنایی کمتر از ۱۰۵؛
* اختلاف صدک ۹۵ و ۲ کمتر از ۴۰؛
* واریانس لاپلاس در ارتفاع تحلیلی ۹۶ پیکسل کمتر از ۶؛
* دامنهٔ میانهٔ تایل‌های روشنایی بیشتر از ۵۰.

این‌ها **heuristic اولیه** هستند و باید با نمونه‌های واقعی کالیبره شوند. فیلد مناسب فقط با اصل خوانده می‌شود. برای فیلد ضعیف یک نسخهٔ جدا با گامای ملایم/CLAHE و در صورت نیاز شارپ‌سازی کم ساخته می‌شود؛ روی هر دو نسخه Paddle امتحان می‌شود، هندسهٔ اصل اولویت دارد و در صورت نبود باکس از نسخهٔ اصلاحی استفاده می‌شود. برش اصل و اصلاحی یک هندسه دارد. بافت زمینه و نام‌های کوتاه می‌توانند روی تشخیص کیفیت اثر بگذارند.

دو خوانش متفاوت → `conflict` و مقدار null. فقط یک خوانش موجود → مقدار با `needs_review`. دو خوانش یکسان → `agreement` در صورت قالب مناسب. قالب مناسبِ عدد یا نام، اثبات صحت آن نیست. تایید دستی یک خوانش مشاهده‌شده، بدون اجرای دوباره:

```python
result.select_variant("national_id", "corrected")
```

تغییر حالت کیفیت:

```python
from persian_ocr import OCRConfig, QualityConfig

ocr = PersianOCR(OCRConfig(quality=QualityConfig(mode="off")))  # only original
# mode="auto" (default) or "force" (compare both for all fields)
```

## محدوده‌ها و انتخاب باکس

قالب پیش‌فرض برای روی کارت با عکس سمت چپ و فیلدها سمت راست است؛ OCR عمومی همهٔ اسناد نیست. مختصات `[left, top, right, bottom]` بین صفر و یک‌اند. مرکز عمودی باکس باید در پنجره باشد و هم‌پوشانی افقی دست‌کم ۳۵٪ عرض خود باکس باشد. عنوان فیلد از قالب می‌آید؛ مقدار از Hezar.

```python
from dataclasses import replace
from persian_ocr import OCRConfig, FieldSpec

base = OCRConfig()
fields = tuple(replace(f, zone=(.45, .32, .84, .42)) if f.key == "first_name" else f
               for f in base.fields)
ocr = PersianOCR(replace(base, fields=fields))

# IDs را از تصویر گزارش همان اجرا بخوانید؛ ترتیب native خود Paddle است.
result = ocr.read("aligned_card.jpg", box_overrides={"first_name": [5]})
# یا {"first_name": {"source": "corrected", "box_ids": [5]}}
```

باکس مشترک چند فیلد رد می‌شود. برش ممکن است به پنجره محدود شود؛ مقدار `clipped_to_template` در جزئیات ثبت می‌شود. باکس چندردیفی وارد CRNN تک‌خطی نمی‌شود. برای چیدمان دیگر پنجره‌ها را تغییر دهید. نمونهٔ JSON تنظیمات در `examples/config.json` است.

## خط فرمان و Colab

```text
persian-ocr aligned_card.jpg --quality auto --json output/data.json --report output/report
persian-ocr camera_photo.jpg --raw-photo --report output/report
persian-ocr aligned_card.jpg --config examples/config.json --detailed
```

stdout یک سند JSON است؛ پیام‌های مدل به stderr هدایت می‌شوند. کد خروج ۲ برای خطای اجرایی است؛ `needs_review` و `not_found` نتیجهٔ ساختاریافته‌اند. خط فرمان با `python -m persian_ocr.cli` هم اجرا می‌شود. برای فایل‌نساختن، گزینه‌های `--json` و `--report` را حذف کنید (کش وزن‌ها مستقل از خروجی‌های کاربر است).

در پوشهٔ `examples` نوت‌بوک `colab_demo.ipynb` قرار دارد. ZIP سورس همین پکیج را آپلود می‌کنید، runtime Python 3.12 جدا می‌سازید و تصویر را می‌دهید. اگر تصویر خام است، نوت‌بوک ZIP CardID را هم می‌گیرد و CUDA را برای استخراج لازم دارد. نتیجهٔ دیکشنری به kernel اصلی برمی‌گردد؛ HTML و PNG هم در صورت درخواست ساخته می‌شوند.

## نگهداری و آزمون

ماژول‌ها: `config`، `images`، `quality`، `geometry`، `backends`، `recognition`، `pipeline`، `result` و `report`. backendها Protocol دارند و برای آزمون قابل تزریق‌اند. یک موتور را برای چند کارت نگه دارید؛ فراخوانی‌های یک نمونه با lock سری می‌شوند. چند موتور همزمان GPU می‌تواند مصرف حافظه را زیاد کند.

```text
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

تست‌های همراه، منطق پردازش و قرارداد adapterها را با backend شبیه‌سازی‌شده می‌سنجند؛ وزن‌های واقعی را دانلود نمی‌کنند. این تحویل شامل سنجش دقت یا تست کامل مدل واقعی روی GPU نیست. نسخه‌های مدل‌ها در pyproject ثبت شده‌اند و snapshot واقعی Hezar پس از بارگذاری در diagnostics ذخیره می‌شود. برای آفلاین Hezar، کش را قبلاً پر کنید؛ برای Paddle هم `detector_model_dir` را به پوشهٔ وزن محلی بدهید و `offline=True` بگذارید.

منابع: [Paddle TextDetection](https://www.paddleocr.ai/v3.3.0/en/version3.x/module_usage/text_detection.html)، [Hezar CRNN](https://huggingface.co/hezarai/crnn-base-fa-v2)، [CLAHE در OpenCV](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html).
