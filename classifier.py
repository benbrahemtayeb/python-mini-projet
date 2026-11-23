from pathlib import Path
from typing import Dict

import joblib

from preprocessing import clean_text

MODELS_DIR = Path(__file__).parent / "models"
_model = None
_vectorizer = None


def load_artifacts():
    """Charge model.pkl et vectorizer.pkl une seule fois (cache en mémoire)."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        model_path = MODELS_DIR / "model.pkl"
        vec_path = MODELS_DIR / "vectorizer.pkl"
        if not model_path.exists() or not vec_path.exists():
            raise FileNotFoundError(
                "Modèle introuvable : lancez d'abord `python train.py`."
            )
        _model = joblib.load(model_path)
        _vectorizer = joblib.load(vec_path)
    return _model, _vectorizer


def predict_sentiment(text: str) -> Dict:
    """Retourne {'sentiment': str, 'confiance': float, 'probabilites': dict}."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Aucun texte exploitable (vide après nettoyage).")
    model, vectorizer = load_artifacts()
    proba = model.predict_proba(vectorizer.transform([cleaned]))[0]
    best = int(proba.argmax())
    return {
        "sentiment": str(model.classes_[best]),
        "confiance": round(float(proba[best]), 4),
        "probabilites": {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)},
    }
