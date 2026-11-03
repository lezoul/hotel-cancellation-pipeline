"""Chargement, audit de qualité et nettoyage de base du jeu Hotel Booking Demand.

Principe : ce module ne fait que des opérations LIGNE PAR LIGNE ou des
suppressions de colonnes qui ne dépendent d'aucune statistique apprise
(pas de moyenne, pas de médiane). Il peut donc être appliqué avant la
séparation entraînement / test sans fuite d'information. Tout ce qui
s'apprend sur les données (imputation, écrêtage, encodage) est fait dans le
pipeline scikit-learn, uniquement sur l'ensemble d'entraînement.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def download_data(path: Path = config.RAW_DATA_PATH, url: str = config.DATA_URL) -> Path:
    """Télécharge hotels.csv depuis le dépôt TidyTuesday s'il est absent."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement de {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut. Les valeurs « NULL » (agent, company) deviennent NaN."""
    path = Path(path)
    if not path.exists():
        download_data(path)
    return pd.read_csv(path, na_values=["NULL", ""], keep_default_na=True)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau de synthèse par colonne : type, valeurs manquantes, cardinalité."""
    report = pd.DataFrame(
        {
            "type": df.dtypes.astype(str),
            "manquants": df.isna().sum(),
            "manquants_%": (df.isna().mean() * 100).round(2),
            "valeurs_uniques": df.nunique(),
        }
    )
    return report.sort_values("manquants_%", ascending=False)


def quality_issues(df: pd.DataFrame) -> pd.Series:
    """Compte les anomalies de qualité repérées lors de l'exploration."""
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    issues = {
        "lignes en double": int(df.duplicated().sum()),
        "réservations sans aucun client (0 adulte, 0 enfant, 0 bébé)": int((guests == 0).sum()),
        "adr négatif": int((df["adr"] < 0).sum()),
        "adr aberrant (> 1 000)": int((df["adr"] > 1000).sum()),
        "séjours de 0 nuit": int(
            ((df["stays_in_weekend_nights"] + df["stays_in_week_nights"]) == 0).sum()
        ),
        "meal = 'Undefined' (équivaut à SC selon la fiche)": int((df["meal"] == "Undefined").sum()),
        "market_segment = 'Undefined'": int((df["market_segment"] == "Undefined").sum()),
        "distribution_channel = 'Undefined'": int(
            (df["distribution_channel"] == "Undefined").sum()
        ),
        "children manquant": int(df["children"].isna().sum()),
        "country manquant": int(df["country"].isna().sum()),
        "agent manquant (pas d'agence)": int(df["agent"].isna().sum()),
        "company manquant (pas d'entreprise)": int(df["company"].isna().sum()),
    }
    return pd.Series(issues, name="nombre de lignes")


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Nettoyage de base sans statistique apprise.

    1. Suppression des colonnes de fuite (statut final, chambre attribuée).
    2. Suppression des doublons exacts : le jeu n'a pas d'identifiant de
       réservation ; garder des copies identiques ferait apparaître les mêmes
       lignes dans l'entraînement ET le test, ce qui gonflerait les scores.
    3. Suppression des réservations sans aucun client et des tarifs impossibles
       (adr négatif ou supérieur à 5 000, erreur de saisie isolée).

    L'année d'arrivée est conservée dans le DataFrame (utile pour les analyses
    et la simulation de surréservation) mais n'est PAS utilisée comme variable
    explicative : elle est absente de config.RAW_INPUT_FEATURES.
    """
    n_start = len(df)
    df = df.drop(columns=config.LEAKAGE_COLUMNS)

    n_before = len(df)
    df = df.drop_duplicates()
    n_dup = n_before - len(df)

    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    mask_no_guest = guests == 0
    mask_bad_adr = (df["adr"] < 0) | (df["adr"] > 5000)
    df = df.loc[~(mask_no_guest | mask_bad_adr)].reset_index(drop=True)

    if verbose:
        print(f"Lignes initiales                     : {n_start:>7,}".replace(",", " "))
        print(f"Doublons supprimés                   : {n_dup:>7,}".replace(",", " "))
        print(f"Réservations sans client supprimées  : {int(mask_no_guest.sum()):>7,}".replace(",", " "))
        print(f"Tarifs impossibles supprimés         : {int(mask_bad_adr.sum()):>7,}".replace(",", " "))
        print(f"Lignes finales                       : {len(df):>7,}".replace(",", " "))
    return df


def split_features_target(df: pd.DataFrame):
    """Sépare X (variables explicatives brutes) et y (is_canceled)."""
    X = df[config.RAW_INPUT_FEATURES].copy()
    y = df[config.TARGET].astype(int).copy()
    return X, y


def train_test_data(X: pd.DataFrame, y: pd.Series):
    """Division 80 / 20 stratifiée sur la cible pour conserver le taux d'annulation."""
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
