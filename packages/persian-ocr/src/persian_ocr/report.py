"""Portable HTML/JSON/PNG report with actual detector boxes and field outcomes."""
import base64
from html import escape
from io import BytesIO

from PIL import ImageDraw

COLORS = {"success": (15, 160, 60), "review": (240, 155, 0), "unused": (135, 135, 135)}


def _inline(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def annotate(result, variant):
    image = result.images[variant].copy()
    draw = ImageDraw.Draw(image)
    assignments = {}
    for key, item in result.fields.items():
        selection = item.get("selection", {})
        if selection.get("geometry_source") == variant:
            for bid in selection["box_ids"]:
                assignments.setdefault(bid, []).append((key, item))
    for box in result.detections.get(variant, {}).get("boxes", []):
        owners = assignments.get(box["id"], [])
        good = bool(owners) and all(item["status"] in ("ok", "agreement", "user_selected")
                                    for _, item in owners)
        color = COLORS["success"] if good else COLORS["review"] if owners else COLORS["unused"]
        polygon = [tuple(p) for p in box["polygon"]]
        draw.line(polygon + [polygon[0]], fill=color, width=3)
        x, y = polygon[0]
        label = str(box["id"]) + (" " + ",".join(k for k, _ in owners) if owners else "")
        draw.text((max(0, x), max(0, y-14)), label, fill=color)
    return image


def save_report(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    files = {"data": result.save_json(directory / "data.json"),
             "details": result.save_json(directory / "details.json", detailed=True)}
    parts = ['<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>گزارش OCR فارسی</title><style>body{font:16px Tahoma,sans-serif;line-height:1.9;margin:24px}',
             'table{border-collapse:collapse;width:100%;margin-bottom:22px}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}',
             'img{max-width:100%;height:auto}.review{background:#fff3d0}pre{white-space:pre-wrap}</style>',
             '<h1>گزارش Paddle → Hezar</h1><p>وضعیت: ' + escape(result.status) + '</p>',
             '<p>سبز: باکس انتخاب‌شده با خوانش پذیرفته‌شده؛ نارنجی: انتخاب‌شده و نیازمند بررسی؛ خاکستری: استفاده‌نشده. رنگ سبز تضمین صحت OCR نیست.</p>']
    if result.diagnostics.get("extraction_needs_review"):
        parts.append('<p class="review">OCR اجرا شده، اما استخراج اولیهٔ کارت نیازمند بررسی بوده است: '
                     + escape(str(result.diagnostics.get("extraction_reason") or 'needs_review')) + '</p>')
    for variant in ("original", "corrected"):
        if variant not in result.images:
            continue
        marked = annotate(result, variant)
        image_path = directory / f"{variant}_boxes.png"
        marked.save(image_path)
        files[variant + "_image"] = image_path
        parts.append('<h2>' + escape(variant) + '</h2><img alt="باکس‌های Paddle" src="' + _inline(marked) + '">')
    parts.append('<table><tr><th>فیلد</th><th>اصل</th><th>اصلاح‌شده</th><th>مقدار</th><th>وضعیت</th></tr>')
    for key, item in result.fields.items():
        direction = "rtl" if item["kind"] == "text" else "ltr"
        parts.append('<tr' + (' class="review"' if item["status"] not in ("ok", "agreement", "user_selected") else '') + '><td>' + escape(item["label"]) + '</td>')
        for value in (item.get("candidates", {}).get("original", {}).get("value"),
                      item.get("candidates", {}).get("corrected", {}).get("value"), item.get("value")):
            parts.append('<td><bdi dir="' + direction + '">' + escape(value or '—') + '</bdi></td>')
        parts.append('<td>' + escape(item["status"]) + '<br>' + escape(item.get("reason", "")) + '</td></tr>')
    parts.append('</table>')
    for key, item in result.fields.items():
        parts.append('<h2>' + escape(item["label"]) + '</h2>')
        quality = item.get("quality", {})
        if quality:
            parts.append('<p>علت اولیهٔ اصلاح: ' + escape(', '.join(quality["reasons"]) or '—') + '</p>')
        for variant, candidate in item.get("candidates", {}).items():
            image_key = candidate.get("image_key")
            parts.append('<h3>' + escape(variant) + '</h3>')
            if image_key in result.images:
                parts.append('<img alt="برش فیلد" src="' + _inline(result.images[image_key]) + '">')
            parts.append('<pre>' + escape(candidate.get("text_raw", candidate.get("error", ""))) + '</pre>')
    parts.append('</html>')
    html_path = directory / "report.html"
    html_path.write_text(''.join(parts), encoding="utf-8")
    files["html"] = html_path
    return files
