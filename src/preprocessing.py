"""Pipelines de prétraitement (Partie 2) et intégration des transformateurs (Partie 3).

Architecture
------------
                        ┌─ continues  : SimpleImputer(median) -> OutlierCapper -> StandardScaler
HotelFeatureEngineer ─> ├─ discrètes  : SimpleImputer(median) -> StandardScaler
   (Partie 3)           ├─ binaires   : SimpleImputer(most_frequent)
                        └─ catégories : SimpleImputer(constant "Inconnu")
                                        -> RareCategoryGrouper -> OneHotEncoder

Justification des choix
-----------------------
* Médiane plutôt que moyenne : lead_time, adr, etc. sont très asymétriques ;
  la médiane n'est pas tirée par les valeurs extrêmes.
* Écrêtage seulement sur les variables continues asymétriques : sur les
  comptages (annulations passées, places de stationnement, jours en liste
  d'attente), le 99e centile vaut 0 ou 1 et l'écrêtage effacerait des
  profils rares mais informatifs.
* StandardScaler : nécessaire pour la régression logistique (référence) et
  pour comparer équitablement les coefficients ; sans effet négatif sur les arbres.
* Binaires laissées en 0/1 : la mise à l'échelle n'apporte rien.
* Catégories : « Inconnu » plutôt que la modalité la plus fréquente pour
  country, car l'absence de pays est elle-même une information. Puis
  regroupement des modalités rares et encodage one-hot (aucun ordre entre
  hôtels, pays, segments). handle_unknown="ignore" protège la prédiction si
  une nouvelle modalité apparaît.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.transformers import HotelFeatureEngineer, OutlierCapper, RareCategoryGrouper


def build_continuous_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "capper",
                OutlierCapper(
                    lower_quantile=config.CAP_LOWER_QUANTILE,
                    upper_quantile=config.CAP_UPPER_QUANTILE,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )


def build_discrete_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_binary_pipeline() -> Pipeline:
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])


def build_categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Inconnu")),
            ("rare_grouper", RareCategoryGrouper(min_freq=config.RARE_CATEGORY_MIN_FREQ)),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_column_transformer() -> ColumnTransformer:
    """Applique le bon traitement à chaque groupe de variables (Partie 2.3)."""
    return ColumnTransformer(
        transformers=[
            ("cont", build_continuous_pipeline(), config.CONTINUOUS_FEATURES),
            ("disc", build_discrete_pipeline(), config.DISCRETE_FEATURES),
            ("bin", build_binary_pipeline(), config.BINARY_FEATURES),
            ("cat", build_categorical_pipeline(), config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessor() -> Pipeline:
    """Prétraitement complet : ingénierie des variables puis ColumnTransformer."""
    return Pipeline(
        steps=[
            ("feature_engineering", HotelFeatureEngineer()),
            ("columns", build_column_transformer()),
        ]
    )
