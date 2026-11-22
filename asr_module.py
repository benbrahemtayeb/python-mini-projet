"""Module Audio : transcription (ASR) avec faster-whisper.

faster-whisper = Whisper d'OpenAI réimplémenté (plus léger, lit mp3/wav sans
installer ffmpeg). Le modèle est téléchargé automatiquement au 1er lancement.
"""
import os
import tempfile

WHISPER_SIZE = os.getenv("WHISPER_MODEL", "small")  # tiny / base / small / medium
_model = None


def get_model():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper n'est pas installé.") from exc
        _model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _model


def audio_path_to_text(path: str) -> str:
    segments, _info = get_model().transcribe(path, language="fr", beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


def audio_bytes_to_text(data: bytes, suffix: str = ".wav") -> str:
    """Écrit les octets dans un fichier temporaire puis transcrit."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()  # fermé avant lecture (nécessaire sous Windows)
        return audio_path_to_text(tmp.name)
    finally:
        os.remove(tmp.name)


def predict_from_audio_path(path: str) -> dict:
    """Audio -> ASR -> classifieur texte (Étape 2, module audio)."""
    from classifier import predict_sentiment
    text = audio_path_to_text(path)
    return {"texte_extrait": text, **predict_sentiment(text)}
