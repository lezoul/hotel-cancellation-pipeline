"""Évaluation du modèle : métriques, graphiques et importance des variables (Partie 5.3)."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config

# Libellés des deux classes, dans l'ordre 0 puis 1
LABELS = ["Maintenue (0)", "Annulée (1)"]


# ---------------------------------------------------------------------------
# Métrique de sélection : F2 maximal atteignable
# ---------------------------------------------------------------------------
def best_fbeta(y_true, y_proba, beta: float = 2) -> float:
    """F-beta maximal atteignable en parcourant tous les seuils possibles.

    Le F2 donne deux fois plus de poids au rappel qu'à la précision. Comme il
    dépend du seuil de décision, on prend sa valeur au MEILLEUR seuil de
    chaque modèle : la comparaison est ainsi équitable entre un modèle avec
    pondération des classes (probabilités gonflées) et un modèle sans.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    b2 = beta**2
    denom = b2 * precision + recall  # formule : (1 + b²) P R / (b² P + R)
    # np.divide avec where : évite la division par zéro quand P = R = 0
    fbeta = np.divide((1 + b2) * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(fbeta.max())


# Scorer utilisable par cross_validate et RandomizedSearchCV.
# response_method="predict_proba" : il reçoit les probabilités, pas les classes.
best_f2_scorer = make_scorer(best_fbeta, response_method="predict_proba", beta=2)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Toutes les métriques utiles pour un seuil de décision donné."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)  # probabilité -> classe 0/1
    return {
        "seuil": threshold,
        "accuracy": accuracy_score(y_true, y_pred),          # part de bonnes prédictions
        "precision": precision_score(y_true, y_pred),        # part des alertes qui sont de vraies annulations
        "recall": recall_score(y_true, y_pred),              # part des annulations détectées
        "f1": f1_score(y_true, y_pred),                      # équilibre précision / rappel
        "f2": fbeta_score(y_true, y_pred, beta=2),           # rappel deux fois plus important
        "roc_auc": roc_auc_score(y_true, y_proba),           # qualité du classement (sans seuil)
        "pr_auc": average_precision_score(y_true, y_proba),  # idem, adaptée au déséquilibre
    }


def report_table(y_true, y_pred) -> pd.DataFrame:
    """Rapport de classification de scikit-learn sous forme de tableau."""
    rep = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
    return pd.DataFrame(rep).T.round(3)


def save_figure(fig, name: str) -> None:
    """Enregistre une figure dans reports/figures (réutilisée dans les diapositives)."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")


def plot_evaluation(y_true, y_proba, threshold: float = 0.5, model_name: str = "Modèle", save: bool = True):
    """Matrice de confusion, courbe ROC et courbe précision-rappel côte à côte."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Matrice de confusion : lignes = réalité, colonnes = prédiction
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=LABELS, cmap="Blues", values_format="d", ax=axes[0], colorbar=False
    )
    axes[0].set_title(f"Matrice de confusion (seuil = {threshold:.2f})")
    axes[0].grid(False)  # le style seaborn ajoute une grille qui masquerait les cases
    axes[0].set_xlabel("Classe prédite")
    axes[0].set_ylabel("Classe réelle")

    # 2. Courbe ROC : taux de vrais positifs en fonction du taux de faux positifs
    RocCurveDisplay.from_predictions(y_true, y_proba, name=model_name, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="Hasard")
    axes[1].set_title("Courbe ROC")
    axes[1].set_xlabel("Taux de faux positifs")
    axes[1].set_ylabel("Taux de vrais positifs (rappel)")
    axes[1].legend(loc="lower right")

    # 3. Courbe précision-rappel : plus parlante quand les classes sont déséquilibrées
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=model_name, ax=axes[2])
    axes[2].axhline(float(np.mean(y_true)), ls="--", color="grey", label="Taux d'annulation")
    axes[2].set_title("Courbe précision-rappel")
    axes[2].set_xlabel("Rappel")
    axes[2].set_ylabel("Précision")
    axes[2].legend(loc="lower left")

    fig.tight_layout()
    if save:
        save_figure(fig, "evaluation_modele.png")
    return fig


def plot_model_comparison(results: pd.DataFrame, save: bool = True):
    """Diagramme en barres des métriques moyennes de validation croisée, par modèle.

    results : une ligne par modèle, colonnes f2_max, pr_auc, roc_auc, precision, recall.
    """
    data = results[["f2_max", "pr_auc", "roc_auc", "precision", "recall"]].copy()
    data.columns = ["F2 maximal", "PR-AUC", "ROC-AUC", "Précision (0,5)", "Rappel (0,5)"]
    # Format « long » (une ligne par modèle et par métrique), attendu par seaborn
    long = data.reset_index().melt(id_vars="modèle", var_name="métrique", value_name="score")
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=long, x="métrique", y="score", hue="modèle", ax=ax)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Comparaison des modèles (validation croisée 5 plis sur l'entraînement)")
    for container in ax.containers:
        ax.bar_label(cast(BarContainer, container), fmt="%.3f", fontsize=7)  # valeur au-dessus de chaque barre
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    if save:
        save_figure(fig, "comparaison_modeles.png")
    return fig


def plot_thresholds(table: pd.DataFrame, t_f2: float, t_f05: float, save: bool = True):
    """Précision, rappel, F2 et F0,5 selon le seuil, avec les deux seuils retenus."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(table["seuil"], table["precision"], label="Précision", color="#2a6f97")
    ax.plot(table["seuil"], table["rappel"], label="Rappel", color="#c44536")
    ax.plot(table["seuil"], table["f2"], label="F2", color="#6a4c93", ls="--")
    ax.plot(table["seuil"], table["f0.5"], label="F0,5", color="#2a9d8f", ls="--")
    ax.axvline(t_f2, color="#6a4c93", ls=":", label=f"seuil F2 = {t_f2:.2f}")
    ax.axvline(t_f05, color="#2a9d8f", ls=":", label=f"seuil F0,5 = {t_f05:.2f}")
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Score")
    ax.set_title("Compromis précision / rappel selon le seuil (prédictions hors pli, entraînement)")
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    fig.tight_layout()
    if save:
        save_figure(fig, "seuils_fbeta.png")
    return fig


def permutation_importance_table(model, X, y, n_samples: int = 5000, n_repeats: int = 5) -> pd.DataFrame:
    """Importance par permutation sur les variables BRUTES (lisible métier).

    On mélange une variable d'origine à la fois et on mesure la baisse de
    PR-AUC du modèle complet : plus la baisse est forte, plus la variable
    compte pour la prédiction.
    """
    if len(X) > n_samples:
        X = X.sample(n_samples, random_state=config.RANDOM_STATE)  # échantillon pour aller plus vite
        y = y.loc[X.index]
    result = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, n_jobs=1,
    )
    return (
        pd.DataFrame(
            {"variable": X.columns, "baisse_pr_auc": np.asarray(result["importances_mean"]),
             "écart": np.asarray(result["importances_std"])}
        )
        .sort_values("baisse_pr_auc", ascending=False)
        .reset_index(drop=True)
    )


def plot_permutation_importance(table: pd.DataFrame, top: int = 15, save: bool = True):
    """Barres horizontales des variables les plus importantes."""
    data = table.head(top).iloc[::-1]  # ordre inversé : la plus importante en haut
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["variable"], data["baisse_pr_auc"], xerr=data["écart"], color="#2a6f97")
    ax.set_xlabel("Baisse moyenne de PR-AUC quand la variable est mélangée")
    ax.set_title(f"Importance par permutation ({top} variables principales)")
    fig.tight_layout()
    if save:
        save_figure(fig, "importance_permutation.png")
    return fig


# ---------------------------------------------------------------------------
# Bilan de la sélection des caractéristiques (Partie 4)
# ---------------------------------------------------------------------------
def selection_report(selector, feature_names) -> pd.DataFrame:
    """Tableau : colonne, importance, retenue ou éliminée (trié par importance).

    selector : SelectFromModel déjà entraîné (étape "selection" du pipeline)
    """
    report = pd.DataFrame(
        {
            "variable": np.asarray(feature_names),
            "importance": selector.estimator_.feature_importances_,  # importance de Gini
            "retenue": selector.get_support(),                       # True si au-dessus du seuil
        }
    )
    report["statut"] = np.where(report["retenue"], "retenue", "éliminée")
    return report.sort_values("importance", ascending=False).reset_index(drop=True)


def original_variable(encoded_name: str, categorical_columns) -> str:
    """Variable d'origine d'une colonne encodée : 'cat__country_PRT' -> 'country'."""
    name = encoded_name.split("__", 1)[-1]  # retire le préfixe de branche (« cat__ »)
    # Noms les plus longs d'abord : évite qu'un nom court soit reconnu dans un nom plus long
    for cat in sorted(categorical_columns, key=len, reverse=True):
        if name.startswith(cat + "_"):
            return cat
    return name


def summarize_by_original_variable(report: pd.DataFrame, categorical_columns) -> pd.DataFrame:
    """Agrège le bilan de sélection par variable d'origine (une ligne par variable brute)."""
    tmp = report.assign(
        variable_origine=report["variable"].map(lambda n: original_variable(str(n), categorical_columns))
    )
    summary = (
        tmp.groupby("variable_origine")
        .agg(
            importance_totale=("importance", "sum"),
            colonnes=("variable", "count"),
            colonnes_retenues=("retenue", "sum"),
        )
        .sort_values("importance_totale", ascending=False)
    )
    summary["statut"] = np.select(
        [summary["colonnes_retenues"] == 0, summary["colonnes_retenues"] == summary["colonnes"]],
        ["éliminée", "retenue"],
        default="partiellement retenue",
    )
    return summary


def mutual_information_ranking(X_transformed: pd.DataFrame, y, sample_size: int = 20000) -> pd.Series:
    """Information mutuelle entre chaque colonne prétraitée et la cible (méthode de contrôle)."""
    from sklearn.feature_selection import mutual_info_classif

    if len(X_transformed) > sample_size:  # échantillon pour limiter le temps de calcul
        idx = np.random.default_rng(config.RANDOM_STATE).choice(len(X_transformed), sample_size, replace=False)
        X_s, y_s = X_transformed.iloc[idx], np.asarray(y)[idx]
    else:
        X_s, y_s = X_transformed, np.asarray(y)
    mi = mutual_info_classif(X_s, y_s, random_state=config.RANDOM_STATE)
    return pd.Series(mi, index=X_transformed.columns, name="information_mutuelle").sort_values(ascending=False)
