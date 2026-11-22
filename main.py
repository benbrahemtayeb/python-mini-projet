"""Étape 3 : API REST FastAPI.

Lancer :  uvicorn main:app --reload
Docs   :  http://127.0.0.1:8000/docs
"""
import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

import asr_module as asr
import classifier
import ocr_module as ocr
from schemas import AudioMeta, ImageMeta, PredictionResponse, TextRequest

app = FastAPI(
    title="Classifieur de Sentiment Multi-source",
    description="Texte, image (OCR) et audio (ASR) -> sentiment positif / négatif / neutre.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ helpers
def _classify(source: str, texte: str) -> PredictionResponse:
    """Brique commune : texte -> sentiment -> réponse JSON unifiée."""
    try:
        res = classifier.predict_sentiment(texte)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PredictionResponse(
        source=source,
        texte_extrait=texte,
        sentiment_predit=res["sentiment"],
        confiance=res["confiance"],
        probabilites=res["probabilites"],
    )


async def _read_and_validate(file: UploadFile, meta_cls) -> bytes:
    """Lit le fichier et le valide avec un schéma Pydantic (MIME, extension, taille)."""
    data = await file.read()
    try:
        meta_cls(filename=file.filename or "", content_type=file.content_type or "", size=len(data))
    except ValidationError as exc:
        msgs = [e["msg"].replace("Value error, ", "") for e in exc.errors()]
        raise HTTPException(status_code=422, detail=msgs)
    return data


async def _handle_text(texte: str) -> PredictionResponse:
    try:
        payload = TextRequest(texte=texte)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=[e["msg"] for e in exc.errors()])
    return _classify("texte", payload.texte)


async def _handle_image(file: UploadFile) -> PredictionResponse:
    data = await _read_and_validate(file, ImageMeta)
    try:  # vérifie que c'est une vraie image (pas juste un nom en .png)
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="Fichier image invalide ou corrompu.")
    try:
        texte = await run_in_threadpool(ocr.image_bytes_to_text, data)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Service OCR indisponible : {exc}")
    return _classify("image", texte)


async def _handle_audio(file: UploadFile) -> PredictionResponse:
    data = await _read_and_validate(file, AudioMeta)
    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    try:
        texte = await run_in_threadpool(asr.audio_bytes_to_text, data, suffix)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Service ASR indisponible : {exc}")
    except Exception:
        raise HTTPException(status_code=422, detail="Fichier audio illisible ou corrompu.")
    return _classify("audio", texte)


# ------------------------------------------------------------------- routes
@app.get("/", tags=["infos"])
def root():
    return {"message": "API Sentiment Multi-source", "docs": "/docs"}


@app.get("/health", tags=["infos"])
def health():
    return {"status": "ok", "ocr_disponible": ocr.is_available()}


@app.post("/predict/text", response_model=PredictionResponse, tags=["prédiction"])
async def predict_text(body: TextRequest):
    """Reçoit un JSON `{"texte": "..."}`."""
    return _classify("texte", body.texte)


@app.post("/predict/image", response_model=PredictionResponse, tags=["prédiction"])
async def predict_image(file: UploadFile = File(..., description="PNG / JPG, 5 Mo max")):
    """Image -> OCR (pytesseract) -> modèle texte."""
    return await _handle_image(file)


@app.post("/predict/audio", response_model=PredictionResponse, tags=["prédiction"])
async def predict_audio(file: UploadFile = File(..., description="WAV / MP3, 20 Mo max")):
    """Audio -> transcription (Whisper) -> modèle texte."""
    return await _handle_audio(file)


@app.post("/predict", response_model=PredictionResponse, tags=["prédiction"])
async def predict_auto(
    texte: Optional[str] = Form(None, description="Texte brut (multipart/form-data)"),
    file: Optional[UploadFile] = File(None, description="Image, audio ou .txt"),
):
    """Détection automatique : texte (champ `texte`) ou fichier (image / audio / .txt)."""
    if file is None and texte is None:
        raise HTTPException(status_code=422, detail="Fournissez un champ 'texte' ou un fichier 'file'.")
    if file is None:
        return await _handle_text(texte)

    ctype = (file.content_type or "").lower()
    ext = Path(file.filename or "").suffix.lower()
    if ctype.startswith("image/") or ext in {".png", ".jpg", ".jpeg"}:
        return await _handle_image(file)
    if ctype.startswith("audio/") or ext in {".wav", ".mp3"}:
        return await _handle_audio(file)
    if ctype == "text/plain" or ext == ".txt":
        raw = await file.read()
        try:
            return await _handle_text(raw.decode("utf-8"))
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="Fichier texte non UTF-8.")
    raise HTTPException(status_code=422, detail=f"Type de fichier non reconnu : '{ctype or ext}'.")
