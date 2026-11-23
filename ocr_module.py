import io
import os
import shutil

import pytesseract
from PIL import Image, ImageOps

OCR_LANG = os.getenv("OCR_LANG", "fra")  


def _configure_tesseract() -> None:
    """Sous Windows tesseract n'est souvent pas dans le PATH : on le cherche."""
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        return
    if shutil.which("tesseract"):
        return
    for path in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                 "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


_configure_tesseract()


def is_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def image_to_text(image: Image.Image) -> str:
    """Extrait le texte brut d'une image PIL."""
    img = ImageOps.grayscale(image.convert("RGB"))
    try:
        raw = pytesseract.image_to_string(img, lang=OCR_LANG, config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract n'est pas installé ou introuvable (voir README)."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise RuntimeError(f"Erreur Tesseract (langue '{OCR_LANG}' installée ?) : {exc}") from exc
    return " ".join(raw.split())  # normalise les espaces / retours à la ligne


def image_bytes_to_text(data: bytes) -> str:
    return image_to_text(Image.open(io.BytesIO(data)))


def image_path_to_text(path: str) -> str:
    return image_to_text(Image.open(path))


def predict_from_image_path(path: str) -> dict:
    """Image -> OCR -> classifieur texte (Étape 2, module image)."""
    from classifier import predict_sentiment
    text = image_path_to_text(path)
    return {"texte_extrait": text, **predict_sentiment(text)}
