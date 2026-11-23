import re
import unicodedata

# Stop-words français. On GARDE volontairement les mots porteurs de sens pour
# le sentiment : pas, ne, non, ni, jamais, rien, très, plus, sans, mal, trop...
STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "d", "l", "et", "ou",
    "a", "au", "aux", "en", "dans", "sur", "sous", "par", "pour", "avec",
    "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
    "je", "j", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "m", "te", "t", "se", "s", "qui", "que", "qu", "dont", "ou",
    "est", "es", "sont", "suis", "ete", "etre", "ai", "as", "avons", "avez",
    "ont", "c", "ca", "cela", "y", "i", "n",
}


def strip_accents(text: str) -> str:
    """é -> e, à -> a ... (rend le modèle robuste aux erreurs OCR/ASR d'accents)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def clean_text(text: str) -> str:
    """minuscules -> sans accents -> sans ponctuation/chiffres -> sans stop-words."""
    text = str(text).lower()
    text = strip_accents(text)
    text = re.sub(r"[^a-z\s]", " ", text)          # garde uniquement lettres/espaces
    tokens = [t for t in text.split() if t not in STOPWORDS_FR and len(t) > 1]
    return " ".join(tokens)
