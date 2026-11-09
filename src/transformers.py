"""Transformateurs personnalisés (Partie 3), sur le modèle du cours (NormaliseurMinMax).

Ce qu'on écrit soi-même (« méthodes à programmer » du cours)
------------------------------------------------------------
- __init__(...)    : mémorise les paramètres, sans calcul
- fit(X, y=None)   : apprend ce qui doit l'être sur l'entraînement, retourne self
- transform(X)     : copie X, applique la règle, retourne la copie

Ce qui est fourni par scikit-learn (outils natifs, non réécrits)
---------------------------------------------------------------
- TransformerMixin      : fit_transform() = fit() puis transform()
- BaseEstimator         : get_params() / set_params(), indispensables pour clone(),
                          la validation croisée et RandomizedSearchCV
- OneToOneFeatureMixin  : get_feature_names_out() pour un transformateur qui garde
                          les mêmes colonnes (noms conservés dans la sortie pandas)
- validate_data()       : appelé dans fit(), enregistre les noms et le nombre de
                          colonnes d'entrée (feature_names_in_, n_features_in_)

Seule exception : HotelFeatureEngineer CRÉE de nouvelles colonnes. Aucun outil
natif ne peut deviner leurs noms, donc get_feature_names_out() y est écrit à la main.

Ces classes sont utilisées comme étapes dans src/pipelines.py.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import validate_data

# Correspondance mois -> saison (saisons météorologiques, hémisphère nord)
SAISON_PAR_MOIS = {
    "December": "Hiver", "January": "Hiver", "February": "Hiver",
    "March": "Printemps", "April": "Printemps", "May": "Printemps",
    "June": "Ete", "July": "Ete", "August": "Ete",
    "September": "Automne", "October": "Automne", "November": "Automne",
}


# ===========================================================================
# 1. HotelFeatureEngineer : création de variables métier
# ===========================================================================
class HotelFeatureEngineer(TransformerMixin, BaseEstimator):
    """Crée des variables métier et corrige les modalités « Undefined ».

    Variables créées
        total_nights    = nuits de semaine + nuits de fin de semaine
        total_guests    = adultes + enfants + bébés
        adr_per_person  = tarif journalier / nombre de clients
        has_children    = 1 si au moins un enfant ou un bébé
        has_agent       = 1 si réservation via une agence (agent renseigné)
        has_company     = 1 si séjour payé par une entreprise (company renseigné)
        arrival_season  = saison déduite du mois d'arrivée
    Corrections
        meal « Undefined » -> « SC » (équivalents selon la fiche des données)
        market_segment, distribution_channel « Undefined » -> valeur manquante
    Colonnes retirées
        agent, company : identifiants sans ordre ; seule leur présence compte

    Comme SalaireAnnuel dans le cours, fit() n'apprend aucune statistique :
    la règle est fixe.
    """

    # Attributs remplis dans fit() par l'outil natif validate_data() ;
    # déclarés ici pour que l'éditeur (VS Code) les connaisse.
    feature_names_in_: np.ndarray
    n_features_in_: int

    def __init__(self, supprimer_identifiants=True):
        # Paramètre : retirer ou non agent et company après création des indicateurs
        self.supprimer_identifiants = supprimer_identifiants

    def fit(self, X, y=None):
        # Outil natif : mémorise les noms et le nombre de colonnes d'entrée
        validate_data(self, X, dtype=None, ensure_all_finite=False)
        return self  # retourne l'objet lui-même pour l'intégration dans un pipeline

    def transform(self, X):
        X_ = pd.DataFrame(X).copy()  # copie pour ne pas modifier les données originales

        # --- Corrections des modalités « Undefined » ---
        X_["meal"] = X_["meal"].replace("Undefined", "SC")
        X_["market_segment"] = X_["market_segment"].replace("Undefined", np.nan)
        X_["distribution_channel"] = X_["distribution_channel"].replace("Undefined", np.nan)

        # --- Nombre de clients (un enfant manquant compte pour 0) ---
        enfants = X_["children"].fillna(0)
        X_["total_guests"] = X_["adults"] + enfants + X_["babies"]

        # --- Nouvelles variables ---
        X_["total_nights"] = X_["stays_in_weekend_nights"] + X_["stays_in_week_nights"]
        # clip(lower=1) évite la division par zéro sans perdre la ligne
        X_["adr_per_person"] = X_["adr"] / X_["total_guests"].clip(lower=1)
        X_["has_children"] = ((enfants + X_["babies"]) > 0).astype(int)  # True/False -> 1/0
        X_["has_agent"] = X_["agent"].notna().astype(int)
        X_["has_company"] = X_["company"].notna().astype(int)
        X_["arrival_season"] = X_["arrival_date_month"].map(SAISON_PAR_MOIS)

        # --- Retrait des identifiants, remplacés par les indicateurs ---
        if self.supprimer_identifiants:
            X_ = X_.drop(columns=["agent", "company"])
        return X_

    def get_feature_names_out(self, input_features=None):
        # Seule méthode non native : les noms des colonnes CRÉÉES ne peuvent pas
        # être devinés par scikit-learn. Sortie = colonnes d'entrée - identifiants + créées.
        colonnes = list(self.feature_names_in_)
        if self.supprimer_identifiants:
            colonnes = [c for c in colonnes if c not in ("agent", "company")]
        creees = ["total_guests", "total_nights", "adr_per_person",
                  "has_children", "has_agent", "has_company", "arrival_season"]
        return np.array(colonnes + creees, dtype=object)


# ===========================================================================
# 2. RareCategoryGrouper : regroupement des modalités rares
# ===========================================================================
class RareCategoryGrouper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Regroupe dans « Autre » les modalités trop peu fréquentes.

    country compte 177 pays dont 157 pèsent chacun moins de 0,5 % des
    réservations : un encodage one-hot direct créerait des dizaines de
    colonnes presque vides. Les modalités fréquentes sont APPRISES dans fit()
    (sur l'entraînement) ; toute autre modalité, y compris une modalité jamais
    vue, devient « Autre » dans transform().
    """

    # Attributs remplis dans fit() par l'outil natif validate_data() ;
    # déclarés ici pour que l'éditeur (VS Code) les connaisse.
    feature_names_in_: np.ndarray
    n_features_in_: int

    def __init__(self, min_freq=0.005, autre="Autre"):
        self.min_freq = min_freq  # fréquence minimale pour garder une modalité
        self.autre = autre        # libellé de regroupement

    def fit(self, X, y=None):
        validate_data(self, X, dtype=None, ensure_all_finite=False)  # outil natif
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        # Étape d'apprentissage : pour chaque colonne, les modalités assez fréquentes
        self.modalites_frequentes_ = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)  # fréquences relatives
            self.modalites_frequentes_[col] = set(freq[freq >= self.min_freq].index)
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        # Étape de transformation : modalité non apprise -> « Autre »
        for col, garder in self.modalites_frequentes_.items():
            valeurs = X_[col].astype(object)
            # where(condition, autre) : garde la valeur si la condition est vraie
            X_[col] = valeurs.where(valeurs.isin(garder) | valeurs.isna(), self.autre)
        return X_


# ===========================================================================
# 3. OutlierCapper : écrêtage des valeurs extrêmes
# ===========================================================================
class OutlierCapper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Ramène les valeurs extrêmes aux centiles appris (winsorisation).

    lead_time monte à 737 jours et adr a une longue queue. Plutôt que de
    supprimer ces réservations, on les borne : même principe que le
    NormaliseurMinMax du cours, fit() apprend une borne basse et une borne
    haute par colonne, transform() les applique.
    """

    # Attributs remplis dans fit() par l'outil natif validate_data() ;
    # déclarés ici pour que l'éditeur (VS Code) les connaisse.
    feature_names_in_: np.ndarray
    n_features_in_: int

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile  # 1er centile par défaut
        self.upper_quantile = upper_quantile  # 99e centile par défaut

    def fit(self, X, y=None):
        validate_data(self, X, dtype=None, ensure_all_finite=False)  # outil natif
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        # Étape d'apprentissage : bornes par colonne, mémorisées dans min_ et max_
        self.min_ = X.quantile(self.lower_quantile)
        self.max_ = X.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        # Étape de transformation : chaque colonne bornée par SES propres limites
        return X_.clip(lower=self.min_, upper=self.max_, axis=1)
