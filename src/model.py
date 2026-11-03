"""Pipeline complet, comparaison de modèles, optimisation et sauvegarde (Partie 5).

Pipeline final
--------------
    HotelFeatureEngineer -> ColumnTransformer -> SelectFromModel -> Classifieur

Toutes les étapes sont apprises uniquement sur l'entraînement (et sur chaque
pli d'entraînement en validation croisée), ce qui exclut toute fuite.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src import config
from src.feature_selection import build_selector
from src.preprocessing import build_preprocessor

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def candidate_models() -> dict:
    """Trois familles d'algorithmes de complexité croissante.

    * Régression logistique : modèle de référence, linéaire et interprétable.
    * Forêt aléatoire       : non linéaire, robuste, peu sensible aux réglages.
    * HistGradientBoosting  : boosting d'arbres, souvent le plus performant
                              sur des données tabulaires de cette taille.
    Pas de class_weight="balanced" : le déséquilibre est modéré (environ 28 %
    d'annulations après nettoyage) et une pondération des classes fausserait
    les probabilités prédites (elles seraient gonflées). Or l'analyse
    prescriptive a besoin de probabilités fiables (perte espérée,
    surréservation). Le déséquilibre est traité au moment de la décision, par
    le choix d'un seuil adapté aux coûts métier (module prescriptive).
    """
    return {
        "Régression logistique": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        "Forêt aléatoire": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }


def build_full_pipeline(model, selector_threshold="median") -> Pipeline:
    """Pipeline de bout en bout : prétraitement + sélection + modèle (Partie 5.2)."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("selection", build_selector(threshold=selector_threshold)),
            ("model", model),
        ]
    )


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def compare_models(X_train, y_train, models: dict | None = None) -> pd.DataFrame:
    """Validation croisée stratifiée de chaque modèle sur l'entraînement seulement."""
    models = models or candidate_models()
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            build_full_pipeline(model),
            X_train,
            y_train,
            cv=cv_splitter(),
            scoring=SCORING,
            n_jobs=1,
        )
        row = {"modèle": name}
        for metric in SCORING:
            row[f"{metric}_moy"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_écart"] = scores[f"test_{metric}"].std()
        row["temps_entrainement_s"] = scores["fit_time"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("modèle").sort_values("roc_auc_moy", ascending=False)


HGB_PARAM_DISTRIBUTIONS = {
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_iter": [200, 400, 600],
    "model__max_leaf_nodes": [31, 63, 127],
    "model__min_samples_leaf": [20, 50, 100],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}


def tune_model(X_train, y_train, model=None, param_distributions=None, n_iter: int = 12):
    """Recherche aléatoire d'hyperparamètres (ROC-AUC, validation croisée 3 plis)."""
    model = model or HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)
    search = RandomizedSearchCV(
        build_full_pipeline(model),
        param_distributions=param_distributions or HGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def save_artifacts(pipeline, metadata: dict, X_test=None, y_test=None) -> None:
    """Sauvegarde le pipeline entraîné, ses métadonnées et l'ensemble de test."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH, compress=3)
    metadata = {**metadata, "date_entrainement": datetime.now().isoformat(timespec="seconds")}
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=float)
    if X_test is not None and y_test is not None:
        X_test.assign(**{config.TARGET: y_test}).to_csv(config.TEST_SET_PATH, index=False)


def load_pipeline():
    """Charge le pipeline et rétablit la sortie pandas entre les étapes.

    Le pipeline a été entraîné avec set_config(transform_output="pandas") :
    chaque étape reçoit des DataFrame avec noms de colonnes. On rétablit ce
    comportement au chargement, quel que soit le réglage global du programme
    qui l'utilise (application Streamlit, script).
    """
    pipeline = joblib.load(config.MODEL_PATH)
    pipeline.set_output(transform="pandas")
    return pipeline


def load_metadata() -> dict:
    with open(config.METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)
