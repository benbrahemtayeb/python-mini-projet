from pathlib import Path
from typing import ClassVar, Dict, Literal, Optional, Set

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positif", "negatif", "neutre"]


# ---------- Texte -----------------------------------------------------------
class TextRequest(BaseModel):
    texte: str = Field(..., min_length=1, max_length=5000,
                       examples=["Livraison rapide, produit conforme, je recommande !"])

    @field_validator("texte")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le texte ne doit pas être vide.")
        return v


# ---------- Réponse unifiée -------------------------------------------------
class PredictionResponse(BaseModel):
    source: Literal["texte", "image", "audio"]
    texte_extrait: str
    sentiment_predit: Sentiment
    confiance: float = Field(..., ge=0, le=1)
    probabilites: Optional[Dict[str, float]] = None


# ---------- Validation des fichiers (type MIME, extension, taille) ----------
class _FileMeta(BaseModel):
    filename: str = Field(..., min_length=1)
    content_type: str
    size: int = Field(..., gt=0, description="taille en octets")

    ALLOWED_MIMES: ClassVar[Set[str]] = set()
    ALLOWED_EXTS: ClassVar[Set[str]] = set()
    MAX_BYTES: ClassVar[int] = 0

    @field_validator("content_type")
    @classmethod
    def check_mime(cls, v: str) -> str:
        v = (v or "").lower()
        if v not in cls.ALLOWED_MIMES:
            raise ValueError(f"Type MIME '{v}' non supporté. Autorisés : {sorted(cls.ALLOWED_MIMES)}")
        return v

    @field_validator("filename")
    @classmethod
    def check_ext(cls, v: str) -> str:
        ext = Path(v).suffix.lower()
        if ext not in cls.ALLOWED_EXTS:
            raise ValueError(f"Extension '{ext}' non supportée. Autorisées : {sorted(cls.ALLOWED_EXTS)}")
        return v

    @field_validator("size")
    @classmethod
    def check_size(cls, v: int) -> int:
        if v > cls.MAX_BYTES:
            raise ValueError(f"Fichier trop volumineux ({v} octets). Maximum : {cls.MAX_BYTES} octets.")
        return v


class ImageMeta(_FileMeta):
    ALLOWED_MIMES: ClassVar[Set[str]] = {"image/png", "image/jpeg", "image/jpg"}
    ALLOWED_EXTS: ClassVar[Set[str]] = {".png", ".jpg", ".jpeg"}
    MAX_BYTES: ClassVar[int] = 5 * 1024 * 1024  # 5 Mo


class AudioMeta(_FileMeta):
    ALLOWED_MIMES: ClassVar[Set[str]] = {
        "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
        "audio/mpeg", "audio/mp3",
    }
    ALLOWED_EXTS: ClassVar[Set[str]] = {".wav", ".mp3"}
    MAX_BYTES: ClassVar[int] = 20 * 1024 * 1024  # 20 Mo
