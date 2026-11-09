"""Sauvegarde et chargement du pipeline entraîné (utilisés par le notebook et l'application)."""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np

from src import config


def save_artifacts(model, metadata: dict, X_test=None, y_test=None) -> None:
    """Sauvegarde le pipeline final, ses métadonnées (JSON) et l'ensemble de test (CSV)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # compress=3 : fichier plus léger, chargement à peine plus lent
    joblib.dump(model, config.MODEL_PATH, compress=3)
    # Date d'entraînement ajoutée automatiquement aux métadonnées
    metadata = {**metadata, "date_entrainement": datetime.now().isoformat(timespec="seconds")}
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        # default=float : convertit les nombres NumPy, que json ne sait pas écrire
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=float)
    if X_test is not None and y_test is not None:
        # L'application Streamlit s'en sert pour les lots et la surréservation
        X_test.assign(**{config.TARGET: np.asarray(y_test)}).to_csv(config.TEST_SET_PATH, index=False)


def load_pipeline():
    """Charge le pipeline final et rétablit la sortie pandas entre ses étapes.

    Le pipeline a été entraîné avec set_config(transform_output="pandas") :
    chaque étape reçoit des DataFrame avec noms de colonnes. On rétablit ce
    comportement au chargement, quel que soit le réglage du programme qui
    l'utilise.
    """
    model = joblib.load(config.MODEL_PATH)
    # Si le seuil a été réglé (TunedThresholdClassifierCV), le pipeline est dans .estimator_
    inner = getattr(model, "estimator_", model)
    inner.set_output(transform="pandas")
    return model


def load_metadata() -> dict:
    """Lit les métadonnées enregistrées avec le pipeline."""
    with open(config.METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)
