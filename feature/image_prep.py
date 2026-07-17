"""In-memory flowchart image preprocessing (never overwrites the input file).

Goal: hand the VLM a clean, legible, right-sized image so it reads node text and
arrows correctly, while capping resolution so we do not blow the T4 memory budget
on visual tokens.

Pipeline: flatten -> auto-crop + pad -> upscale-if-small -> mild contrast ->
resolution cap. Optional OCR augmentation returns the detected text so the
prompt can list it alongside the image.

Only Pillow is required. OpenCV enables optional CLAHE/deskew; Tesseract or
PaddleOCR enable optional OCR. Everything degrades gracefully if a dependency
is missing.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, ImageFilter

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False

# Tunables (kept local; the token cap is enforced again by the Qwen processor).
_PAD = 24
_MIN_SIDE = 1000
_MAX_LONG_SIDE = 1600


def _flatten_to_rgb(img: Image.Image) -> Image.Image:
    """RGBA / palette -> RGB composited on white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def _autocrop(img: Image.Image, pad: int = _PAD) -> Image.Image:
    """Trim uniform (near-white) margins, then re-pad with white."""
    gray = ImageOps.grayscale(img)
    # Anything darker than near-white is "ink". Invert so getbbox finds it.
    mask = gray.point(lambda p: 255 if p < 245 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return img
    cropped = img.crop(bbox)
    return ImageOps.expand(cropped, border=pad, fill=(255, 255, 255))


def _upscale_if_small(img: Image.Image, min_side: int = _MIN_SIDE) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    if short >= min_side:
        return img
    scale = min_side / float(short)
    # Do not explode a very wide/tall chart past the long-side cap.
    if max(w, h) * scale > _MAX_LONG_SIDE:
        scale = _MAX_LONG_SIDE / float(max(w, h))
    if scale <= 1.0:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _cap_long_side(img: Image.Image, max_long: int = _MAX_LONG_SIDE) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_long:
        return img
    scale = max_long / float(max(w, h))
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _enhance_legibility(img: Image.Image) -> Image.Image:
    """Mild, non-destructive: autocontrast + light unsharp. Deliberately NOT
    binarized (hard thresholds eat thin arrowheads). Kept as 3-channel RGB."""
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=2))
    return gray.convert("RGB")


def preprocess_flowchart(path: str | Path) -> Image.Image:
    """Full pipeline. Returns an RGB PIL.Image ready for the Qwen processor."""
    img = Image.open(path)
    img = _flatten_to_rgb(img)
    img = _autocrop(img)
    img = _upscale_if_small(img)
    img = _cap_long_side(img)
    img = _enhance_legibility(img)
    return img


# --------------------------------------------------------------------------- #
# Optional OCR augmentation
# --------------------------------------------------------------------------- #
def _ocr_tesseract(img: Image.Image) -> str:
    import pytesseract  # type: ignore

    # eng + tha covers English keywords and Thai labels used in the exercises.
    try:
        return pytesseract.image_to_string(img, lang="eng+tha")
    except Exception:
        return pytesseract.image_to_string(img)


def _ocr_paddle(img: Image.Image) -> str:
    from paddleocr import PaddleOCR  # type: ignore

    global _PADDLE
    try:
        _PADDLE
    except NameError:
        _PADDLE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    if not _HAS_NUMPY:
        return ""
    result = _PADDLE.ocr(np.array(img), cls=True)
    lines = []
    for page in result or []:
        for _box, (text, _conf) in page or []:
            lines.append(text)
    return "\n".join(lines)


def ocr_flowchart(img: Image.Image) -> str:
    """Best-effort text extraction. Returns "" if no OCR engine is available."""
    for engine in (_ocr_tesseract, _ocr_paddle):
        try:
            text = engine(img)
            if text and text.strip():
                return text.strip()
        except Exception:
            continue
    return ""
