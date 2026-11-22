# Classifieur de Sentiment Multi-source (E-commerce Customer Voice)

Texte, image (OCR) et audio (ASR) sont convertis en texte, puis un classifieur
unique (TF-IDF + Régression Logistique) prédit : positif / négatif / neutre.

## Installation
1. Python 3.10+ et **Tesseract OCR** avec les données françaises (`fra`).
   - Windows : installeur UB-Mannheim (cocher "French" dans les langues).
   - Linux : `sudo apt install tesseract-ocr tesseract-ocr-fra`
2. Dépendances :
```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux : source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation
```bash
python train.py --augment                 # entraîne et crée models/model.pkl + vectorizer.pkl
uvicorn main:app --reload                 # API  -> http://127.0.0.1:8000/docs
streamlit run streamlit_app.py            # mini-app web (API démarrée)
pytest -v                                 # tests automatisés
python evaluate_multimodal.py             # évalue OCR + ASR sur le dataset
```

## Variables d'environnement (optionnelles)
| Variable | Rôle | Défaut |
|---|---|---|
| `TESSERACT_CMD` | chemin de tesseract.exe | auto-détecté |
| `OCR_LANG` | langue OCR | `fra` |
| `WHISPER_MODEL` | tiny / base / small / medium | `small` |

## Endpoints
| Route | Entrée |
|---|---|
| `POST /predict/text` | JSON `{"texte": "..."}` |
| `POST /predict/image` | multipart `file` (png/jpg, 5 Mo max) |
| `POST /predict/audio` | multipart `file` (wav/mp3, 20 Mo max) |
| `POST /predict` | multipart : champ `texte` OU `file` (type détecté) |

Réponse :
```json
{"source": "texte", "texte_extrait": "...", "sentiment_predit": "positif", "confiance": 0.89}
```

## Captures d'écran
(à ajouter : Streamlit + Swagger /docs)

## Choix techniques
- TF-IDF (1-2 grammes) + Régression Logistique : léger, rapide, probabilités pour la confiance.
- Nettoyage : minuscules, accents retirés, stop-words (les négations sont conservées).
- OCR : pytesseract + Pillow (niveaux de gris, `--psm 6`).
- ASR : faster-whisper (Whisper optimisé CPU), langue forcée en français.
- Validation Pydantic : type MIME, extension et taille des fichiers.
