# استخراج کارت و تشخیص متن با PaddleOCR + خواندن فارسی با Hezar — نسخه ۰٫۲

مسیر پیش‌فرض:
DINO + SAM → هندسه و جهت قبلی → PP-OCRv5_server_det → Hezar CRNN → شش فیلد.
EasyOCR فقط برای تعیین جهت در استخراج باقی است. Hezar CRAFT در مسیر پیش‌فرض لود نمی‌شود.

## اجرای Colab
۱. ZIP تازه مخزن را دانلود کنید یا ZIP مستقل پروژه را بسازید.
۲. DINO_SAM_Hezar_Colab.ipynb را در Colab باز و T4 GPU را فعال کنید.
۳. سلول اول هر دو نوع ZIP را می‌پذیرد. سلول نصب را اجرا کنید؛ دو محیط مستقل ساخته می‌شود.
۴. عکس‌ها را آپلود و پردازش را اجرا کنید. جدول شش‌ردیفی و خروجی‌های تصویری/JSON/CSV در بخش نمایش هستند.

PADDLE_DEVICE پیش‌فرض cpu است؛ فقط مدل تشخیص متن روی CPU است. DINO/SAM، EasyOCR و Hezar روی GPU باقی می‌مانند. این انتخاب صریح برای نصب قابل‌اتکاتر بدون تداخل CUDA است و fallback پنهان نیست. gpu:0 گزینه نصب اختیاری Paddle CUDA 12.6 است؛ درایور مناسب و دسترسی به مخزن رسمی لازم دارد و در این محیط تست نشده است.

TEXT_DETECTOR پیش‌فرض paddle است. برای مقایسه با مسیر قبلی craft را انتخاب کنید.
وزن Paddle با نام و revision ثابت دانلود می‌شود. جهت و سیاست هندسی استخراج تغییر نکرده است؛ needs_review همچنان قبل از OCR متوقف می‌شود.

## جداسازی محیط‌ها
اسکریپت نصب محیط Torch/Hezar و محیط Paddle را جدا نگه می‌دارد؛ اولی opencv-python-headless و دومی opencv-contrib-python دارد. دو توزیع cv2 نباید در یک محیط نصب شوند. worker مدل Paddle را یک بار لود می‌کند و عکس‌ها را از pipe حافظه‌ای می‌گیرد؛ عکس برای سرویس بیرونی ارسال نمی‌شود. JSON فقط مختصات/امتیاز را برمی‌گرداند. timeout و خطای مدل صریح‌اند؛ OCR جایگزین خودکار نمی‌شود. بعد از استفاده pipeline.close() را صدا بزنید؛ پایان پردازش هم worker را متوقف می‌کند.

## API
```python
from persian_id_ocr import PersianCardOCR
pipeline = PersianCardOCR(cache_dir='./models')
try:
    result = pipeline.process('card.jpg')
    print(result.text)
    print(result.to_dataframe())
finally:
    pipeline.close()
```

مسیر worker از فایل paddle-worker.json داخل محیط اصلی پیدا می‌شود. در نصب سفارشی PERSIAN_PADDLE_PYTHON را به Python محیط مستقل Paddle تنظیم کنید.

## خروجی‌ها و محدوده اعتبار
regions شامل box، source_polygons، detection_scores، روش برش و متن خام/نرمال‌شده است؛ مختصات همگی به کارت صاف‌شده مربوط‌اند. چندضلعی تک‌ناحیه‌ای با پرسپکتیو صاف و حاشیه زمینه آماده CRNN می‌شود. اتصال قطعه‌های CRAFT برای Paddle خاموش است تا عنوان و مقدار ناخواسته یکی نشوند.

fields.csv همیشه شش ردیف دارد: national_id، first_name، last_name، date_of_birth، father_name، expiry_date. مقدار ناموجود خالی می‌ماند. انتساب بر اساس قالب مکانی کارت استاندارد است، نه مدل آموزش‌دیده. اگر Paddle یک کادر شامل عنوان و مقدار بدهد، ممکن است فیلد missing شود؛ هیچ متنی با پاسخ CSV یا حدس تصحیح نمی‌شود. candidate تأیید صحت نیست.

روی دو کارت صاف‌شده کاربر و یک تصویر سفید با وزن‌های واقعی Paddle و Hezar روی CPU آزمایش شده است. این تست شامل DINO/SAM یا اجرای کامل CUDA نبوده و بهتر بودن از CRAFT را ثابت نمی‌کند. جزئیات در VALIDATION.md است.

## بسته مستقل
```bash
python scripts/build_zip.py --output dist/dino-sam-hezar-colab.zip
```

## منابع
- https://github.com/PaddlePaddle/PaddleOCR/blob/v3.3.2/docs/version3.x/module_usage/text_detection.en.md
- https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det
- https://huggingface.co/hezarai/crnn-base-fa-v2
