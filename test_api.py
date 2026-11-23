import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient  # TestClient de FastAPI (basé sur httpx)
from PIL import Image

import asr_module
import classifier
import main
import ocr_module
from preprocessing import clean_text

client = TestClient(main.app)
SENTIMENTS = {"positif", "negatif", "neutre"}
DATASET = Path("data_test_v2")


# ----------------------------------------------------------------- utilitaires
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (80, 40), "white").save(buf, format="PNG")
    return buf.getvalue()


def check_structure(body: dict, source: str):
    assert set(["source", "texte_extrait", "sentiment_predit", "confiance"]) <= set(body)
    assert body["source"] == source
    assert body["sentiment_predit"] in SENTIMENTS
    assert 0 <= body["confiance"] <= 1
    assert isinstance(body["texte_extrait"], str) and body["texte_extrait"]


@pytest.fixture
def fake_ocr(monkeypatch):
    monkeypatch.setattr(ocr_module, "image_bytes_to_text",
                        lambda data: "Livraison express et produit parfait, je recommande !")


@pytest.fixture
def fake_asr(monkeypatch):
    monkeypatch.setattr(asr_module, "audio_bytes_to_text",
                        lambda data, suffix=".wav": "Très mécontent, le produit ne fonctionne pas du tout.")


# ------------------------------------------------------------ tests unitaires
def test_clean_text():
    out = clean_text("Très DÉÇU !!! Le colis #123")
    assert out == "tres decu colis"


def test_predict_sentiment_structure():
    res = classifier.predict_sentiment("Excellent produit, je recommande vivement")
    assert res["sentiment"] in SENTIMENTS
    assert 0 <= res["confiance"] <= 1
    assert abs(sum(res["probabilites"].values()) - 1) < 1e-3


def test_predict_sentiment_empty_raises():
    with pytest.raises(ValueError):
        classifier.predict_sentiment("!!! ???")


# --------------------------------------------------------------------- infos
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


# ------------------------------------------------------------ /predict/text
def test_predict_text_ok():
    r = client.post("/predict/text", json={"texte": "Excellente qualité, très satisfait, je recommande !"})
    assert r.status_code == 200
    check_structure(r.json(), "texte")


def test_predict_text_clear_cases():
    pos = client.post("/predict/text", json={"texte": "Excellente qualité, très satisfait, bravo !"}).json()
    neg = client.post("/predict/text", json={"texte": "Article cassé, très déçu, remboursement jamais reçu."}).json()
    assert pos["sentiment_predit"] == "positif"
    assert neg["sentiment_predit"] == "negatif"


@pytest.mark.parametrize("payload", [
    {},                       # champ manquant
    {"texte": ""},            # vide
    {"texte": "     "},       # que des espaces
    {"texte": 123},           # mauvais type
    {"texte": "a" * 5001},    # trop long
    {"text": "mauvais nom"},  # mauvaise clé
])
def test_predict_text_invalid_payload(payload):
    assert client.post("/predict/text", json=payload).status_code == 422


def test_predict_text_not_json():
    r = client.post("/predict/text", content="pas du json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_predict_text_unusable_text():
    r = client.post("/predict/text", json={"texte": "!!! ???"})
    assert r.status_code == 422  # vide après nettoyage


# ----------------------------------------------------------- /predict/image
def test_predict_image_ok(fake_ocr):
    r = client.post("/predict/image", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    check_structure(r.json(), "image")


def test_predict_image_wrong_mime():
    r = client.post("/predict/image", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_predict_image_corrupted():
    r = client.post("/predict/image", files={"file": ("a.png", b"pas une image", "image/png")})
    assert r.status_code == 422


def test_predict_image_empty_file():
    r = client.post("/predict/image", files={"file": ("a.png", b"", "image/png")})
    assert r.status_code == 422


def test_predict_image_too_large():
    big = b"0" * (5 * 1024 * 1024 + 1)
    r = client.post("/predict/image", files={"file": ("a.png", big, "image/png")})
    assert r.status_code == 422


def test_predict_image_missing_file():
    assert client.post("/predict/image").status_code == 422


@pytest.mark.skipif(not ocr_module.is_available(), reason="Tesseract non installé")
def test_predict_image_real_ocr():
    """Test d'intégration avec le vrai OCR sur une image du dataset."""
    path = DATASET / "images" / "I02.png"
    with open(path, "rb") as f:
        r = client.post("/predict/image", files={"file": ("I02.png", f.read(), "image/png")})
    assert r.status_code == 200
    check_structure(r.json(), "image")


# ----------------------------------------------------------- /predict/audio
def test_predict_audio_ok(fake_asr):
    r = client.post("/predict/audio", files={"file": ("a.wav", b"RIFF....", "audio/wav")})
    assert r.status_code == 200
    check_structure(r.json(), "audio")


def test_predict_audio_mp3_ok(fake_asr):
    r = client.post("/predict/audio", files={"file": ("a.mp3", b"ID3....", "audio/mpeg")})
    assert r.status_code == 200


def test_predict_audio_wrong_mime():
    r = client.post("/predict/audio", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 422


def test_predict_audio_wrong_extension():
    r = client.post("/predict/audio", files={"file": ("a.exe", b"xxxx", "audio/wav")})
    assert r.status_code == 422


def test_predict_audio_too_large():
    big = b"0" * (20 * 1024 * 1024 + 1)
    r = client.post("/predict/audio", files={"file": ("a.wav", big, "audio/wav")})
    assert r.status_code == 422


# ------------------------------------------------------------------ /predict
def test_auto_text():
    r = client.post("/predict", data={"texte": "Produit correct pour le prix, sans plus."})
    assert r.status_code == 200
    check_structure(r.json(), "texte")


def test_auto_image(fake_ocr):
    r = client.post("/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    check_structure(r.json(), "image")


def test_auto_audio(fake_asr):
    r = client.post("/predict", files={"file": ("a.mp3", b"ID3....", "audio/mpeg")})
    assert r.status_code == 200
    check_structure(r.json(), "audio")


def test_auto_txt_file():
    r = client.post("/predict", files={"file": ("avis.txt", "Très satisfait, bravo !".encode(), "text/plain")})
    assert r.status_code == 200
    check_structure(r.json(), "texte")


def test_auto_nothing_sent():
    assert client.post("/predict").status_code == 422


def test_auto_unsupported_type():
    r = client.post("/predict", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 422
