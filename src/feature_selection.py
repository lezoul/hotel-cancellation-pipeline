"""Sélection des caractéristiques (Partie 4).

Méthode retenue : SelectFromModel avec une forêt aléatoire (méthode intégrée,
« embedded »).

Pourquoi cette méthode pour ce jeu de données
---------------------------------------------
* Après l'encodage one-hot, on obtient une centaine de colonnes de natures
  différentes (continues standardisées, binaires, indicatrices de pays...).
  Les importances d'une forêt se calculent de la même façon pour tous ces types.
* Les relations avec l'annulation sont non linéaires et comportent des
  interactions (ex. délai de réservation x type de dépôt x segment) ; une
  forêt les capte, contrairement à un filtre univarié (chi2, ANOVA) ou à
  une corrélation linéaire.
* La sélection est une étape du pipeline : elle est réapprise à chaque pli
  de validation croisée, donc sans fuite vers les données de validation.
* Coût raisonnable sur ~87 000 lignes, contrairement à RFE qui réentraîne
  le modèle des dizaines de fois.

Seuil : « median » garde les variables dont l'importance dépasse la médiane,
soit environ la moitié. Le notebook vérifie que cette réduction ne dégrade
pas les performances.

Une méthode de contrôle (information mutuelle, filtre univarié non linéaire)
est fournie pour confronter le classement dans le notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif

from src import config


def build_selector(threshold: str | float = "median") -> SelectFromModel:
    estimator = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    return SelectFromModel(estimator=estimator, threshold=threshold)


def selection_report(selector: SelectFromModel, feature_names) -> pd.DataFrame:
    """Tableau : variable, importance, retenue ou éliminée (trié par importance)."""
    importances = selector.estimator_.feature_importances_
    report = pd.DataFrame(
        {
            "variable": np.asarray(feature_names),
            "importance": importances,
            "retenue": selector.get_support(),
        }
    )
    report["statut"] = np.where(report["retenue"], "retenue", "éliminée")
    return report.sort_values("importance", ascending=False).reset_index(drop=True)


def original_variable(encoded_name: str) -> str:
    """Retrouve la variable d'origine à partir du nom encodé.

    Exemple : 'cat__country_PRT' -> 'country', 'cont__lead_time' -> 'lead_time'.
    """
    name = encoded_name.split("__", 1)[-1]
    for cat in sorted(config.CATEGORICAL_FEATURES, key=len, reverse=True):
        if name.startswith(cat + "_"):
            return cat
    return name


def summarize_by_original_variable(report: pd.DataFrame) -> pd.DataFrame:
    """Agrège la sélection par variable d'origine (une ligne par variable brute)."""
    tmp = report.assign(variable_origine=report["variable"].map(original_variable))
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
    """Information mutuelle entre chaque colonne prétraitée et la cible (contrôle)."""
    if len(X_transformed) > sample_size:
        idx = np.random.default_rng(config.RANDOM_STATE).choice(
            len(X_transformed), sample_size, replace=False
        )
        X_s, y_s = X_transformed.iloc[idx], np.asarray(y)[idx]
    else:
        X_s, y_s = X_transformed, np.asarray(y)
    mi = mutual_info_classif(X_s, y_s, random_state=config.RANDOM_STATE)
    return pd.Series(mi, index=X_transformed.columns, name="information_mutuelle").sort_values(
        ascending=False
    )
