"""Entraînement complet en ligne de commande.

Reproduit, sans le notebook, la chaîne qui produit le modèle utilisé par
l'application Streamlit :

    python -m src.train            # entraînement avec optimisation des hyperparamètres
    python -m src.train --fast     # sans recherche d'hyperparamètres (plus rapide)

Étapes
------
1. Chargement et nettoyage (doublons, lignes impossibles, colonnes de fuite)
2. Séparation 80 / 20 stratifiée
3. Recherche d'hyperparamètres (HistGradientBoosting) en validation croisée
4. Choix du seuil de décision sur des prédictions hors pli de l'entraînement
5. Évaluation finale sur le test, sauvegarde du pipeline et des métadonnées
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn import set_config
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

from src import config
from src import data_loader as dl
from src.evaluation import compute_metrics
from src.feature_selection import selection_report
from src.model import build_full_pipeline, cv_splitter, save_artifacts, tune_model
from src.prescriptive import optimal_threshold, revenue_at_risk

# Hypothèses de coûts par défaut pour le seuil de décision (€)
DEFAULT_RESALE_RATE = 0.3     # part des chambres annulées que l'hôtel revend
DEFAULT_ACTION_EFFICACY = 0.35  # part des annulations évitées par l'action (prépaiement partiel)
DEFAULT_COST_FP = 30.0        # action inutile : 5 € de coût direct + 25 € de perte de bienveillance client


def main(fast: bool = False) -> dict:
    set_config(transform_output="pandas")

    df = dl.clean_data(dl.load_raw_data())
    X, y = dl.split_features_target(df)
    X_train, X_test, y_train, y_test = dl.train_test_data(X, y)

    if fast:
        best_params = {}
        pipeline = build_full_pipeline(
            HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)
        )
    else:
        print("Recherche d'hyperparamètres ...")
        search = tune_model(X_train, y_train)
        best_params = search.best_params_
        print(f"Meilleurs paramètres : {best_params}  (ROC-AUC CV = {search.best_score_:.4f})")
        pipeline = search.best_estimator_

    # Seuil de décision choisi SANS regarder le test : prédictions hors pli
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv_splitter(), method="predict_proba")[:, 1]
    nights = X_train["stays_in_weekend_nights"] + X_train["stays_in_week_nights"]
    # Faux négatif : annulation non détectée = revenu qu'une action aurait pu sauver
    cost_fn = float(
        np.mean(revenue_at_risk(X_train["adr"], nights, DEFAULT_RESALE_RATE)) * DEFAULT_ACTION_EFFICACY
    )
    threshold, _ = optimal_threshold(y_train, oof_proba, cost_fn=cost_fn, cost_fp=DEFAULT_COST_FP)

    pipeline.fit(X_train, y_train)
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    metrics_05 = compute_metrics(y_test, proba_test, 0.5)
    metrics_opt = compute_metrics(y_test, proba_test, threshold)

    selector = pipeline.named_steps["selection"]
    feat_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    report = selection_report(selector, feat_names)

    metadata = {
        "modele": type(pipeline.named_steps["model"]).__name__,
        "hyperparametres": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in best_params.items()},
        "n_entrainement": int(len(X_train)),
        "n_test": int(len(X_test)),
        "taux_annulation": float(y.mean()),
        "seuil_optimal": threshold,
        "hypotheses_seuil": {
            "taux_revente": DEFAULT_RESALE_RATE,
            "efficacite_action": DEFAULT_ACTION_EFFICACY,
            "cout_faux_negatif": round(cost_fn, 2),
            "cout_faux_positif": DEFAULT_COST_FP,
        },
        "metriques_test_seuil_0_5": metrics_05,
        "metriques_test_seuil_optimal": metrics_opt,
        "variables_retenues": report.loc[report["retenue"], "variable"].tolist(),
        "variables_eliminees": report.loc[~report["retenue"], "variable"].tolist(),
        "valeurs_reference": {
            "countries": sorted(X_train["country"].dropna().unique().tolist()),
            "adr_median": float(X_train["adr"].median()),
        },
    }

    test_out = X_test.copy()
    test_out["arrival_date_year"] = df.loc[X_test.index, "arrival_date_year"].values
    save_artifacts(pipeline, metadata, test_out, y_test)

    print(f"Seuil optimal (coûts) : {threshold:.2f}")
    print("Test, seuil 0,5     :", {k: round(v, 4) for k, v in metrics_05.items()})
    print("Test, seuil optimal :", {k: round(v, 4) for k, v in metrics_opt.items()})
    print(f"Pipeline sauvegardé dans {config.MODEL_PATH}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne le pipeline de prédiction des annulations.")
    parser.add_argument("--fast", action="store_true", help="sans recherche d'hyperparamètres")
    main(fast=parser.parse_args().fast)
