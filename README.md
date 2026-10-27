# Persian OCR ID Card

پروژه محلی با Python 3.11 و محیط مجازی مستقل `.venv`.
عکس در سیستم محلی پردازش می‌شود. دانلود اولیه وزن‌ها و داده به اینترنت نیاز دارد.

## شروع

در ترمینال:

```bash
cd /Users/neginnoori/Documents/Codex/Persian_OCR_IDCard
source .venv/bin/activate
jupyter lab notebooks/Persian_Card_OCR.ipynb
```

یا فایل `start_jupyter.command` را اجرا کنید. کرنل Notebook باید **Persian Card OCR (.venv)** باشد.
بخش نصب Notebook در این پروژه لازم نیست؛ وابستگی‌ها از قبل در محیط پروژه نصب می‌شوند.

## عکس محلی

```bash
persian-card-ocr --image /path/to/card.jpg --output outputs/my_card
```

## داده Kaggle

```bash
persian-card-ocr --download-dataset
persian-card-ocr --dataset-id 354 --output outputs/card_354
```

در صورت درخواست احراز هویت Kaggle، در Notebook یک بار `kagglehub.login()` را اجرا کنید.
نسخه CLI مسیر کش واقعی Kaggle را استفاده می‌کند و به مسیرهای Colab وابسته نیست.

## مسیر پردازش و خروجی

DocAligner ONNX چهار گوشه پیشنهادی را تعیین می‌کند؛ OpenCV پرسپکتیو را اصلاح می‌کند؛
دو جهت افقی با امتیاز ابتکاری OCR مقایسه می‌شوند؛ Hezar CRAFT نواحی متن و CRNN متن فارسی را می‌خواند.
در Mac این نسخه از CPU استفاده می‌کند؛ CUDA مخصوص GPU انویدیا است.

پوشه خروجی شامل `boundary.jpg`, `rectified.png`, `oriented.png`, `text_boxes.png`,
برش‌ها، `text.txt` و `result.json` است. در Notebook جدول CSV و ZIP هم ساخته می‌شود.
نام‌گذاری شش فیلد هویتی در این نسخه پیاده نشده است؛ خروجی متن OCR است.

`ocr_candidate` به معنی صحت قطعی نیست. `orientation_uncertain` نیاز به بررسی جهت دارد.
کنترل هندسی هویت شیء را تأیید نمی‌کند؛ کارت ناقص یا گوشه پوشیده ممکن است تشخیص اشتباه داشته باشد.
متن خام حفظ می‌شود؛ هیچ رقمی با حدس یا با پاسخ CSV اصلاح نمی‌شود.

## منابع

- https://github.com/DocsaidLab/DocAligner
- https://huggingface.co/hezarai/CRAFT
- https://huggingface.co/hezarai/crnn-base-fa-v2

ایده‌های مخزن پیشنهادی کاربر بررسی شده‌اند، اما کدی از آن کپی نشده است:
https://github.com/Sherviiiin/iranian-id-card-ocr/tree/master/cv_proj

## بررسی

```bash
python -m unittest discover -s tests -v
```

نسخه‌های نصب‌شده در `requirements-lock.txt` ثبت می‌شوند. این فایل وضعیت محیط همین سیستم است.

## وضعیت اجرای اولیه

نصب وابستگی‌ها، واردکردن مدل‌های Hezar، ثبت کرنل، بررسی ساختار Notebook و چهار آزمون کد موفق بوده‌اند.
وزن‌های DocAligner و Hezar دانلود شده‌اند. مسیر کامل روی عکس ارسالی `Unknown-2.jpeg` روی CPU اجرا شد؛
زمان پردازش پس از بارگذاری مدل‌ها حدود ۶٫۴ ثانیه بود. کادر و جهت در بررسی تصویری مناسب بودند، اما متن OCR خطا دارد.
این یک اجرای آزمایشی است و ارزیابی دقت دیتاست محسوب نمی‌شود.
نتیجه در `outputs/local_smoke_test` و گزارش بررسی در `verification.json` موجود است.
