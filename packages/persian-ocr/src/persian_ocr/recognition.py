"""Representation normalization, format checks and conservative variant selection."""
import re

DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def interpret(raw, spec):
    if spec.kind == "text":
        value = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", raw)
        value = re.sub(r"\s+", " ", value).strip(" *#`|:：؛،-–")
        # Remove only an exact title with a separator at an edge.
        if value == spec.label:
            value = ""
        elif re.match(re.escape(spec.label) + r"\s*[:：]", value):
            value = re.sub(r"^" + re.escape(spec.label) + r"\s*[:：]\s*", "", value)
        valid = bool(value) and not any(c.isdigit() for c in value)
        check = "nonempty text without digits; not a correctness check"
    else:
        value = re.sub(r"[\s\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", raw.translate(DIGITS))
        if spec.kind == "national_id":
            valid = bool(re.fullmatch(r"[0-9]{10}", value))
            check = "10 digits only; no checksum or identity verification"
        else:
            match = re.fullmatch(r"([0-9]{4})/([0-9]{1,2})/([0-9]{1,2})", value)
            valid = False
            if match:
                year, month, day = map(int, match.groups())
                valid = year > 0 and 1 <= month <= 12 and 1 <= day <= (31 if month <= 6 else 30)
            check = "date format and simple month/day ranges, not full Jalali validation"
    return value or None, {"format_ok": valid, "check": check}


def decide(candidates, override=None):
    if override is not None:
        if override not in candidates or override not in ("original", "corrected"):
            raise ValueError("Chosen variant is unavailable")
        candidate = candidates[override]
        if candidate["status"] != "ok" or not candidate.get("value"):
            raise ValueError("Chosen variant produced no text")
        return {"value": candidate["value"], "status": "user_selected",
                "selected_variant": override, "reason": "Explicit selection after review"}
    original, corrected = candidates.get("original", {}), candidates.get("corrected", {})
    first = original.get("value") if original.get("status") == "ok" else None
    second = corrected.get("value") if corrected.get("status") == "ok" else None
    if "corrected" not in candidates:
        return {"value": first, "status": "ok" if first and original["validation"]["format_ok"] else "needs_review",
                "selected_variant": "original", "reason": "Original-only field"}
    if first and second and first == second:
        return {"value": first, "status": "agreement" if original["validation"]["format_ok"] else "needs_review",
                "selected_variant": "original", "reason": "Both variants agree; not proof of correctness"}
    if first and second:
        return {"value": None, "status": "conflict", "reason": "Different readings; compare variants"}
    if first or second:
        variant = "original" if first else "corrected"
        return {"value": first or second, "status": "needs_review", "selected_variant": variant,
                "reason": "Only one variant returned text"}
    return {"value": None, "status": "needs_review", "reason": "No readable value"}
