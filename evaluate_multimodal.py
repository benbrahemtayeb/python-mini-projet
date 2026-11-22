"""Évalue la chaîne complète OCR/ASR -> classifieur sur le dataset fourni.

Usage :  python evaluate_multimodal.py            (images + audios)
         python evaluate_multimodal.py --no-audio (images seulement, plus rapide)
"""
import argparse
import difflib
from pathlib import Path

import pandas as pd

DATA = Path("data_test_v2")


def find_file(folder: Path, name: str) -> Path:
    """audios.csv parle de .wav mais les fichiers fournis sont des .mp3 -> on cherche par nom."""
    p = folder / name
    if p.exists():
        return p
    matches = list(folder.glob(Path(name).stem + ".*"))
    if not matches:
        raise FileNotFoundError(f"{name} introuvable dans {folder}")
    return matches[0]


def run(kind: str, csv_name: str, ref_col: str, func) -> None:
    df = pd.read_csv(DATA / csv_name)
    folder = DATA / kind
    ok = 0
    print(f"\n===== {kind.upper()} =====")
    for _, row in df.iterrows():
        res = func(str(find_file(folder, row["fichier"])))
        sim = difflib.SequenceMatcher(None, res["texte_extrait"].lower(), row[ref_col].lower()).ratio()
        good = res["sentiment"] == row["sentiment_ref"]
        ok += good
        print(f"{row['fichier']}: attendu={row['sentiment_ref']:8s} prédit={res['sentiment']:8s} "
              f"conf={res['confiance']:.2f} similarité_texte={sim:.2f} {'OK' if good else 'ERREUR'}")
        print(f"    extrait : {res['texte_extrait']}")
    print(f"--> Accuracy {kind} : {ok}/{len(df)} = {ok / len(df):.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-image", action="store_true")
    args = ap.parse_args()
    if not args.no_image:
        from ocr_module import predict_from_image_path
        run("images", "images.csv", "texte_attendu_ocr", predict_from_image_path)
    if not args.no_audio:
        from asr_module import predict_from_audio_path
        run("audios", "audios.csv", "transcription_reference", predict_from_audio_path)
