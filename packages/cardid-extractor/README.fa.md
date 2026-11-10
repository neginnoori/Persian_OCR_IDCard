# پکیج استخراج کارت CardID

ورودی یک عکس است و خروجی **تصویر کارت با پرسپکتیو و جهت اصلاح‌شده**، مشابه سلول تأییدشدهٔ نوت‌بوک.
تصویر خروجی `PIL.Image.Image` با حالت `RGB` و اندازهٔ پیش‌فرض ۸۵۶×۵۴۰ است.

## استفاده در برنامه

```python
from cardid import CardExtractor

# یک بار برای هر process؛ مدل‌ها بین درخواست‌ها در حافظه می‌مانند.
extractor = CardExtractor()

def prepare_card(input_image):
    return extractor.extract(input_image)

card = prepare_card("photo.jpg")
# text = persian_ocr(card)  # تابع همکارتان
```

اگر OCR آرایهٔ RGB می‌خواهد، `numpy.asarray(card)` را به آن بده. پکیج، OCR نهایی فیلدها را انجام نمی‌دهد.
EasyOCR داخلی فقط برای انتخاب جهت کارت است. کتابخانه به‌صورت خودکار تصویر ذخیره یا نمایش نمی‌دهد.
در صورت نیاز خودت `card.save("card.png")` را اجرا کن.

ورودی می‌تواند مسیر فایل، bytes فایل آپلودشده، تصویر PIL یا آرایهٔ NumPy از نوع `uint8` باشد.
آرایهٔ OpenCV را با `extractor.extract(image, color_order="BGR")` بده؛ پیش‌فرض آرایه‌ها RGB است.
grayscale تک‌کاناله پذیرفته و به سه کانال یکسان تبدیل می‌شود. دقت عکس‌های خاکستری باید با نمونه‌های واقعی ارزیابی شود.

## نصب و GPU

Python 3.10 تا 3.13 و NVIDIA GPU لازم است. CUDA GPU شمارهٔ صفر اجباری است؛ روی CPU ادامه نمی‌دهد.
مدل‌های DINO، SAM و EasyOCR روی GPU هستند؛ عملیات OpenCV و خواندن تصویر روی CPU انجام می‌شوند.

برای برنامهٔ اصلی، یک محیط مجازی جدا بساز و کد زیر را از ریشهٔ سورس اجرا کن:

```bash
python -m venv .venv
```

فعال‌سازی در Linux:

```bash
source .venv/bin/activate
```

فعال‌سازی در PowerShell ویندوز:

```powershell
.\.venv\Scripts\Activate.ps1
```

سپس:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-colab.txt .
python -m pip check
```

این پروفایل از CUDA 12.6 استفاده می‌کند و درایور سازگار لازم دارد. برای نصب فقط فایل wheel نیز می‌توان
از `python -m pip install cardid_extractor-0.1.0-py3-none-any.whl` در محیط جدا استفاده کرد؛ ابتدا PyTorch سازگار با CUDA را نصب کن.
فایل requirements نسخه‌های مستقیم را ثابت می‌کند؛ قفل کامل تمام وابستگی‌های غیرمستقیم نیست.

## اجرای Colab

۱. فایل `examples/colab_gpu.ipynb` را باز کن و از Runtime → Change runtime type یک GPU مثل T4 انتخاب کن.

۲. سلول آپلود بسته را اجرا کن و فایل ZIP سورس همین تحویل را انتخاب کن.

۳. سلول نصب، محیط مستقل `/content/cardid-venv` می‌سازد. وابستگی‌های این پکیج در آن نصب می‌شوند؛
نسخهٔ `huggingface-hub` محیط اصلی Colab دست‌کاری نمی‌شود. بنابراین تعارض قبلی با Gradio و Diffusers به محیط این پکیج منتقل نمی‌شود.

۴. عکس‌ها را آپلود کن و سلول اجرای نمونه را اجرا کن. همان شیء استخراج‌کننده برای همهٔ عکس‌های آن اجرا استفاده می‌شود.
در این روش، کد پکیج با Python محیط مستقل اجرا می‌شود؛ kernel اصلی نوت‌بوک عوض نمی‌شود.

۵. خروجی‌ها را ببین و ZIP نتایج را دانلود کن. گزینهٔ تست grayscale هم در نوت‌بوک وجود دارد.

نصب اولیه و دریافت وزن مدل‌ها اینترنت لازم دارد. وزن‌ها داخل بسته نیستند. اولین فراخوانی کندتر است چون مدل‌ها بارگذاری می‌شوند.
برای ماندگاری وزن‌ها در Colab می‌توان پوشه‌ای از Drive را mount کرد و آن را به `cache_dir` داد.

## مدیریت نتیجه و خطا

```python
from cardid import CardExtractor, ExtractorConfig

extractor = CardExtractor(ExtractorConfig(cache_dir="./models"))
extractor.load()  # اختیاری: بارگذاری در شروع سرویس

result = extractor.extract_result(input_image)
if result.ok:
    card = result.image
else:
    print(result.status, result.reason)
```

وضعیت‌ها `ok`، `needs_review` و `not_found` هستند. در حالت مبهم، `result.image` ممکن است فقط یک پیش‌نمایش برای بررسی باشد؛
آن را خودکار به OCR نفرست. تابع سادهٔ `extract` فقط تصویر پذیرفته‌شده را برمی‌گرداند و در بقیهٔ موارد
`CardNotFoundError` یا `ReviewRequiredError` می‌دهد. دومی نتیجه را در `error.result` نگه می‌دارد.
خطاهای ورودی، نبود GPU، بارگذاری مدل و اجرای مدل نیز خطاهای مشخص از خانوادهٔ `CardIDError` هستند.

برای اجرای آفلاین پس از دانلود کامل وزن‌ها:

```python
extractor = CardExtractor(ExtractorConfig(cache_dir="./models", offline=True))
```

پکیج برای **روی کارت ملی ایرانی** و همان چیدمان نوت‌بوک طراحی شده است. تا سه کادر متمایز DINO با چیدمان فیلدها و محل چهره ارزیابی می‌شوند. اگر هیچ‌کدام پذیرفته نشود، پکیج لبه‌های شبیه کارت را بررسی می‌کند. اگر مرز کارت در تصویر پیوسته نباشد، ناحیه‌های روشن و تیرهٔ بزرگ هم بررسی می‌شوند. کادر جایگزین با شواهد کافی از چهره یا چیدمان فیلدها می‌تواند به‌عنوان برش پیشنهادی برگردد. اگر کارت قاب عکس را پر کرده باشد یا کنار آن نوار سفید باشد، با شواهد کافی از چیدمان کارت، برش بر اساس قاب عکس هم برای بازبینی پیشنهاد می‌شود. برش جایگزینِ مماس با مرز تصویر نیاز به بررسی دارد. برش کوچک و مبهمِ بدون شواهد کافی کنار گذاشته می‌شود.
برای پشت کارت و اسناد عمومی تضمینی ندارد. برچسب `ok` هم احتمال آماری یا تضمین صحت صددرصد نیست.

رابط اصلی و مدل‌ها از نوت‌بوک جدا شده‌اند؛ نیازی به matplotlib، pandas، Hezar، KaggleHub یا YOLO ندارد.
یک نمونه برای هر process بساز؛ چند worker مستقل، هرکدام مدل و حافظهٔ GPU خود را دارند.

## آزمایش

```bash
python examples/smoke_gpu.py photo.jpg --also-gray --output-dir results
python -m pip install -e ".[dev]"
python -m pytest
```

فایل `VALIDATION.md` دقیقاً مشخص می‌کند چه آزمایش‌هایی انجام شده‌اند. آزمایش مدل‌های واقعی روی GPU و Colab
با آزمایش‌های شبیه‌سازی‌شدهٔ منطق برنامه فرق دارد؛ موفقیت آن‌ها به معنی تأیید دقت مدل روی مجموعهٔ واقعی نیست.
