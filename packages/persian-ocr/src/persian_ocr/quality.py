"""v13 quality heuristics, refactored to pure in-memory image operations."""
from dataclasses import asdict
import math

import cv2
import numpy as np
from PIL import Image

from .geometry import pixel_zone

def metrics(rgb):
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    # Fixed analysis height makes this heuristic less sensitive to the extraction width.
    scale = min(96 / max(1,gray.shape[0]), 1600 / max(1,gray.shape[1]))
    analysis = cv2.resize(gray, (max(8,round(gray.shape[1]*scale)),max(8,round(gray.shape[0]*scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    # A name can occupy less than 5% of its window; p2 retains sparse dark strokes.
    p2,p50,p95 = np.percentile(gray,[2,50,95])
    tiles = [float(np.median(tile)) for row in np.array_split(gray,3,axis=0)
             for tile in np.array_split(row,4,axis=1) if tile.size]
    return {"median_brightness":float(p50), "contrast_p95_p2":float(p95-p2),
            "laplacian_variance_h96":float(cv2.Laplacian(analysis,cv2.CV_64F).var()),
            "tile_median_range":float(max(tiles)-min(tiles)), "native_size":[gray.shape[1],gray.shape[0]]}


def assess(image, field_specs, quality):
    q = asdict(quality)
    fields = {}
    for spec in field_specs:
        key, label = spec.key, spec.label
        bb = pixel_zone(spec.zone, image.size)
        rect = [max(0,math.floor(bb[0])),max(0,math.floor(bb[1])),min(image.width,math.ceil(bb[2])),min(image.height,math.ceil(bb[3]))]
        crop = image.crop(tuple(rect))
        if min(crop.size) < 8:
            raise ValueError(f"Field window too small for quality analysis: {key}")
        m = metrics(crop)
        reasons = []
        if m["median_brightness"] < q["dark_median"]:
            reasons.append("dark")
        if m["contrast_p95_p2"] < q["low_contrast"]:
            reasons.append("low_contrast")
        if m["laplacian_variance_h96"] < q["blur_laplacian"]:
            reasons.append("soft_edges")
        if m["tile_median_range"] > q["uneven_light"]:
            reasons.append("uneven_light")
        enabled = q["mode"] == "force" or (q["mode"] == "auto" and bool(reasons))
        fields[key] = {"label":label,"metrics":m,"reasons":reasons,"enhance":enabled,
                       "decision":"forced" if q["mode"]=="force" else ("disabled" if q["mode"]=="off" else ("try_correction" if enabled else "original_only"))}
    return {"mode":q["mode"],"fields":fields,"enhance":any(f["enhance"] for f in fields.values()),
            "heuristic_not_calibrated":True,"settings":q}


def correct_card(image, assessment):
    q = assessment["settings"]
    rgb = np.array(image.convert("RGB"))
    active = [f for f in assessment["fields"].values() if f["enhance"]]
    all_reasons = {reason for f in active for reason in f["reasons"]}
    operations = []
    if "dark" in all_reasons or q["mode"]=="force":
        dark_medians = [f["metrics"]["median_brightness"] for f in active if "dark" in f["reasons"]]
        median = float(np.median(dark_medians)) if dark_medians else 125.0
        gamma = float(np.clip(math.log(150/255)/math.log(max(1,median)/255),.65,1.0))
        lut = np.clip(255*(np.arange(256,dtype=float)/255)**gamma,0,255).astype(np.uint8)
        lab = cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.LUT(lab[:,:,0],lut)
        rgb = cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
        operations.append({"operation":"lighten_L_gamma","exponent":gamma})
    if all_reasons & {"dark","low_contrast","uneven_light"} or q["mode"]=="force":
        lab = cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
        l = lab[:,:,0]
        adjusted = cv2.createCLAHE(clipLimit=q["clahe_clip"],tileGridSize=(8,8)).apply(l)
        lab[:,:,0] = cv2.addWeighted(l,1-q["clahe_blend"],adjusted,q["clahe_blend"],0)
        rgb = cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
        operations.append({"operation":"CLAHE_L","clip":q["clahe_clip"],"blend":q["clahe_blend"]})
    # No sharpening of the whole card: apply it only to weak field crops later.
    if not operations:
        operations.append({"operation":"copy_for_field_sharpening_only"})
    return Image.fromarray(rgb),operations


def sharpen_crop(crop, amount):
    image = np.array(crop.convert("RGB"))
    smooth = cv2.GaussianBlur(image,(0,0),.8)
    result = cv2.addWeighted(image,1+amount,smooth,-amount,0)
    return Image.fromarray(result)


