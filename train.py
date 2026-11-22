"""Étape 1 : entraînement du classifieur texte + sauvegarde des artefacts.

Usage :  python train.py             (entraîne sur textes.csv uniquement)
         python train.py --augment   (ajoute aussi les textes de référence OCR/ASR)
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from preprocessing import clean_text

DATA_DIR = Path("data_test_v2")
MODELS_DIR = Path("models")
LABELS = ["positif", "negatif", "neutre"]
RANDOM_STATE = 42


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def make_model() -> LogisticRegression:
    return LogisticRegression(C=10, max_iter=1000)


def load_reference_texts():
    """Textes de référence des images (OCR) et audios (ASR) : (textes, labels)."""
    texts, labels = [], []
    for csv_name, col in [("images.csv", "texte_attendu_ocr"),
                          ("audios.csv", "transcription_reference")]:
        d = pd.read_csv(DATA_DIR / csv_name)
        texts += d[col].tolist()
        labels += d["sentiment_ref"].tolist()
    return texts, labels


def main(augment: bool = False) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # 1) Chargement + nettoyage ------------------------------------------------
    df = pd.read_csv(DATA_DIR / "textes.csv")
    if augment:  # petit dataset (24 textes) -> on ajoute les 18 textes OCR/ASR
        ref_t, ref_l = load_reference_texts()
        df = pd.concat([df, pd.DataFrame({"texte": ref_t, "sentiment_ref": ref_l})],
                       ignore_index=True)
    df["texte_clean"] = df["texte"].apply(clean_text)
    print(f"{len(df)} textes chargés. Répartition :\n{df['sentiment_ref'].value_counts()}\n")
    print("Exemple avant/après nettoyage :")
    print("  AVANT :", df["texte"].iloc[1])
    print("  APRÈS :", df["texte_clean"].iloc[1], "\n")

    X, y = df["texte_clean"], df["sentiment_ref"]

    # 2) Évaluation sur un split train/test stratifié ---------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    vec, clf = make_vectorizer(), make_model()
    clf.fit(vec.fit_transform(X_tr), y_tr)
    y_pred = clf.predict(vec.transform(X_te))

    print("=== Évaluation (split 75/25) ===")
    print("Accuracy :", round(accuracy_score(y_te, y_pred), 3))
    print(classification_report(y_te, y_pred, labels=LABELS, zero_division=0))
    cm = confusion_matrix(y_te, y_pred, labels=LABELS)
    print("Matrice de confusion (lignes = vrai, colonnes = prédit) :\n", cm, "\n")
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues")
    plt.title("Matrice de confusion (jeu de test)")
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Validation croisée (le dataset est petit -> plus fiable qu'un seul split)
    cv_pipe = make_pipeline(make_vectorizer(), make_model())
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(cv_pipe, X, y, cv=skf)
    print(f"=== Validation croisée 4-fold === {scores.round(2)}  moyenne = {scores.mean():.3f}\n")

    # 4) Test de robustesse : textes issus de l'OCR / ASR (jamais vus à l'entraînement)
    if not augment:  # sinon ce serait de la fuite de données (data leakage)
        ref_t, ref_l = load_reference_texts()
        ext_pred = clf.predict(vec.transform([clean_text(t) for t in ref_t]))
        print("=== Robustesse sur textes OCR/ASR de référence (modèle du split) ===")
        print("Accuracy :", round(accuracy_score(ref_l, ext_pred), 3), "\n")

    # 5) Entraînement FINAL sur toutes les données + sauvegarde -----------------
    vec_final, clf_final = make_vectorizer(), make_model()
    clf_final.fit(vec_final.fit_transform(X), y)
    joblib.dump(clf_final, MODELS_DIR / "model.pkl")
    joblib.dump(vec_final, MODELS_DIR / "vectorizer.pkl")
    print("Artefacts sauvegardés : models/model.pkl, models/vectorizer.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true",
                        help="ajoute les textes de référence OCR/ASR au jeu d'entraînement")
    main(parser.parse_args().augment)
