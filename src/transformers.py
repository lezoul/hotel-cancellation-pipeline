"""Transformateurs personnalisés compatibles scikit-learn (Partie 3).

Chaque classe hérite de BaseEstimator (pour get_params / set_params, donc
compatible avec GridSearchCV et clone) et de TransformerMixin (qui fournit
fit_transform). Chaque classe implémente :

- fit(X, y=None)              : apprend ce qui doit l'être sur l'entraînement
- transform(X)                : applique la transformation
- get_feature_names_out(...)  : noms des colonnes produites (traçabilité et
                                compatibilité avec set_output(transform="pandas"))

Trois transformateurs :

1. HotelFeatureEngineer : crée des variables métier (durée totale, nombre de
   clients, saison, présence d'une agence ou d'une entreprise, tarif par
   personne) et corrige les modalités « Undefined ».
2. RareCategoryGrouper  : regroupe les modalités rares (ex. 170+ pays) dans
   « Autre », seuil appris sur l'entraînement.
3. OutlierCapper        : écrête les valeurs extrêmes aux centiles appris sur
   l'entraînement (winsorisation), au lieu de supprimer des lignes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

SEASON_BY_MONTH = {
    "December": "Hiver", "January": "Hiver", "February": "Hiver",
    "March": "Printemps", "April": "Printemps", "May": "Printemps",
    "June": "Ete", "July": "Ete", "August": "Ete",
    "September": "Automne", "October": "Automne", "November": "Automne",
}


def _to_frame(X, columns=None) -> pd.DataFrame:
    """Garantit un DataFrame (le pipeline peut transmettre un tableau NumPy)."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X, columns=columns)


class HotelFeatureEngineer(BaseEstimator, TransformerMixin):
    """Ingénierie des caractéristiques propre au domaine hôtelier.

    Variables créées
    ----------------
    total_nights    : nuits de semaine + nuits de fin de semaine
    total_guests    : adultes + enfants + bébés
    adr_per_person  : tarif journalier moyen divisé par le nombre de clients
    has_children    : 1 si au moins un enfant ou un bébé
    has_agent       : 1 si la réservation passe par une agence (agent renseigné)
    has_company     : 1 si une entreprise paie la réservation (company renseigné)
    arrival_season  : saison d'arrivée déduite du mois

    Corrections
    -----------
    meal « Undefined » -> « SC » (la fiche TidyTuesday indique qu'ils sont équivalents)
    market_segment / distribution_channel « Undefined » -> NaN (imputés ensuite)

    Colonnes retirées
    -----------------
    agent, company : identifiants numériques sans ordre (352 entreprises, 94 %
    de manquants). Seule l'information « présent / absent » est conservée.

    Paramètres
    ----------
    drop_ids : bool, défaut True
        Retirer les identifiants agent et company après création des indicateurs.
    """

    def __init__(self, drop_ids: bool = True):
        self.drop_ids = drop_ids

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _to_frame(X)

        X["meal"] = X["meal"].replace("Undefined", "SC")
        for col in ("market_segment", "distribution_channel"):
            X[col] = X[col].replace("Undefined", np.nan)

        children = pd.to_numeric(X["children"], errors="coerce").fillna(0)
        babies = pd.to_numeric(X["babies"], errors="coerce").fillna(0)
        adults = pd.to_numeric(X["adults"], errors="coerce").fillna(0)

        X["total_nights"] = X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
        X["total_guests"] = adults + children + babies
        # .clip(lower=1) évite la division par zéro sans perdre la ligne
        X["adr_per_person"] = X["adr"] / X["total_guests"].clip(lower=1)
        X["has_children"] = ((children + babies) > 0).astype(int)
        X["has_agent"] = X["agent"].notna().astype(int)
        X["has_company"] = X["company"].notna().astype(int)
        X["arrival_season"] = X["arrival_date_month"].map(SEASON_BY_MONTH)

        if self.drop_ids:
            X = X.drop(columns=["agent", "company"])
        return X

    def fit(self, X, y=None):
        X = _to_frame(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = np.array(self._engineer(X.head(2)).columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        return self._engineer(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Regroupe les modalités peu fréquentes d'une variable catégorielle.

    Pourquoi : country compte plus de 170 pays dont la majorité représente
    moins de 0,5 % des réservations. Un encodage one-hot direct créerait
    des dizaines de colonnes presque vides, sources de bruit et de
    surapprentissage. Les modalités rares sont regroupées dans « Autre ».

    Les modalités conservées sont apprises dans fit (sur l'entraînement
    seulement). Une modalité jamais vue en entraînement devient aussi
    « Autre » au moment de la prédiction, ce qui rend le pipeline robuste.

    Paramètres
    ----------
    min_freq : float, défaut 0.005
        Fréquence relative minimale pour qu'une modalité soit conservée.
    other_label : str, défaut "Autre"
        Libellé de la modalité de regroupement.
    """

    def __init__(self, min_freq: float = 0.005, other_label: str = "Autre"):
        self.min_freq = min_freq
        self.other_label = other_label

    def fit(self, X, y=None):
        X = _to_frame(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.frequent_categories_ = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True, dropna=True)
            self.frequent_categories_[col] = set(freq[freq >= self.min_freq].index)
        return self

    def transform(self, X):
        check_is_fitted(self, "frequent_categories_")
        X = _to_frame(X, columns=self.feature_names_in_)
        for col, keep in self.frequent_categories_.items():
            values = X[col].astype(object)
            X[col] = values.where(values.isin(keep) | values.isna(), self.other_label)
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "frequent_categories_")
        return self.feature_names_in_

    def grouped_summary(self) -> pd.DataFrame:
        """Nombre de modalités conservées par variable (pour le rapport)."""
        check_is_fitted(self, "frequent_categories_")
        return pd.DataFrame(
            {
                "variable": list(self.frequent_categories_),
                "modalites_conservees": [len(v) for v in self.frequent_categories_.values()],
            }
        )


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Écrête (winsorise) les valeurs extrêmes aux centiles appris.

    Pourquoi : lead_time monte à 737 jours, les séjours à 69 nuits et adr
    présente une queue très longue. Plutôt que de supprimer ces lignes
    (perte d'information sur la cible), les valeurs sont ramenées aux bornes
    [q_bas, q_haut] calculées sur l'entraînement. La standardisation qui
    suit n'est ainsi plus dominée par quelques points extrêmes.

    Paramètres
    ----------
    lower_quantile : float, défaut 0.01
    upper_quantile : float, défaut 0.99
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Il faut 0 <= lower_quantile < upper_quantile <= 1.")
        X = _to_frame(X).apply(pd.to_numeric, errors="coerce")
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.lower_bounds_ = X.quantile(self.lower_quantile)
        self.upper_bounds_ = X.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        X = _to_frame(X, columns=self.feature_names_in_).apply(pd.to_numeric, errors="coerce")
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "lower_bounds_")
        return self.feature_names_in_

    def bounds(self) -> pd.DataFrame:
        """Bornes apprises par variable (pour le rapport)."""
        check_is_fitted(self, "lower_bounds_")
        return pd.DataFrame({"borne_basse": self.lower_bounds_, "borne_haute": self.upper_bounds_})
