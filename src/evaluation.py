"""Évaluation du modèle : métriques, graphiques et importance des variables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config

LABELS = ["Maintenue (0)", "Annulée (1)"]


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "seuil": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def report_table(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
    return pd.DataFrame(rep).T.round(3)


def save_figure(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")


def plot_evaluation(y_true, y_proba, threshold: float = 0.5, model_name: str = "Modèle", save: bool = True):
    """Matrice de confusion, courbe ROC et courbe précision-rappel côte à côte."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=LABELS, cmap="Blues", values_format="d", ax=axes[0], colorbar=False
    )
    axes[0].set_title(f"Matrice de confusion (seuil = {threshold:.2f})")
    axes[0].grid(False)
    axes[0].set_xlabel("Classe prédite")
    axes[0].set_ylabel("Classe réelle")

    RocCurveDisplay.from_predictions(y_true, y_proba, name=model_name, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="Hasard")
    axes[1].set_title("Courbe ROC")
    axes[1].set_xlabel("Taux de faux positifs")
    axes[1].set_ylabel("Taux de vrais positifs (rappel)")
    axes[1].legend(loc="lower right")

    PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=model_name, ax=axes[2])
    axes[2].axhline(np.mean(y_true), ls="--", color="grey", label="Taux d'annulation")
    axes[2].set_title("Courbe précision-rappel")
    axes[2].set_xlabel("Rappel")
    axes[2].set_ylabel("Précision")
    axes[2].legend(loc="lower left")

    fig.tight_layout()
    if save:
        save_figure(fig, "evaluation_modele.png")
    return fig


def plot_model_comparison(results: pd.DataFrame, save: bool = True):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    data = results[[f"{m}_moy" for m in metrics]].copy()
    data.columns = metrics
    long = data.reset_index().melt(id_vars="modèle", var_name="métrique", value_name="score")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="métrique", y="score", hue="modèle", ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Comparaison des modèles (validation croisée 5 plis, entraînement)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save:
        save_figure(fig, "comparaison_modeles.png")
    return fig


def permutation_importance_table(pipeline, X, y, n_samples: int = 5000, n_repeats: int = 5) -> pd.DataFrame:
    """Importance par permutation sur les variables BRUTES (lisible métier).

    On mélange une variable d'origine à la fois et on mesure la baisse de
    ROC-AUC du pipeline complet : plus la baisse est forte, plus la variable
    compte pour la prédiction.
    """
    if len(X) > n_samples:
        X = X.sample(n_samples, random_state=config.RANDOM_STATE)
        y = y.loc[X.index]
    result = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=1
    )
    return (
        pd.DataFrame(
            {"variable": X.columns, "baisse_roc_auc": result.importances_mean, "écart": result.importances_std}
        )
        .sort_values("baisse_roc_auc", ascending=False)
        .reset_index(drop=True)
    )


def plot_permutation_importance(table: pd.DataFrame, top: int = 15, save: bool = True):
    data = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["variable"], data["baisse_roc_auc"], xerr=data["écart"], color="#2a6f97")
    ax.set_xlabel("Baisse moyenne de ROC-AUC quand la variable est mélangée")
    ax.set_title(f"Importance par permutation ({top} variables principales)")
    fig.tight_layout()
    if save:
        save_figure(fig, "importance_permutation.png")
    return fig
