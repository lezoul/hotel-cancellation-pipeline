"""Pipelines du projet, construits selon le modèle du cours « Pipelines de prétraitement ».

Ce script est importé dans notebooks/01_pipeline.ipynb :

    from src.pipelines import pipeline_num, pipeline_cat, preprocessor, model_pipeline, ...

Structure (du plus petit au plus grand)
---------------------------------------
1. Sous-pipelines par type de colonnes   (cours : « Préparation de pipelines spécifiques »)
       pipeline_continu, pipeline_discret, pipeline_binaire, pipeline_cat
2. ColumnTransformer                       (cours : « Combine les traitements »)
       preprocessor
3. Pipeline de prétraitement complet       (transformateur personnalisé + ColumnTransformer)
       pipeline_pretraitement
4. Pipeline complet                        (cours : « Prétraitement & Modèle d'apprentissage »)
       model_pipeline = prétraitement -> sélection des caractéristiques -> modèle

Utilisation (cours)
-------------------
    model_pipeline.fit(X_train, y_train)          # fit_transform sur chaque étape, puis fit du modèle
    y_pred = model_pipeline.predict(X_test)       # transform sur chaque étape, puis predict
    model_pipeline.steps[0]                       # accès par index
    model_pipeline.named_steps["model"]           # accès par nom
    model_pipeline.named_steps["preprocessing"].transform(X_train)   # données transformées

Les objets définis ici ne sont pas encore entraînés. Pour entraîner plusieurs
variantes sans les mélanger, le notebook en fait des copies avec
sklearn.base.clone().
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.transformers import HotelFeatureEngineer, OutlierCapper, RareCategoryGrouper

# ===========================================================================
# 0. Colonnes à traiter
#    (après HotelFeatureEngineer, qui crée total_nights, total_guests,
#     adr_per_person, has_children, has_agent, has_company, arrival_season)
# ===========================================================================

# Variables continues très asymétriques : imputation, écrêtage, standardisation
colonnes_continues = [
    "lead_time",                 # délai entre réservation et arrivée (jusqu'à 737 jours)
    "adr",                       # tarif journalier moyen
    "stays_in_weekend_nights",   # nuits de fin de semaine
    "stays_in_week_nights",      # nuits de semaine
    "total_nights",              # créée : nuits totales
    "adr_per_person",            # créée : tarif par personne
]

# Variables discrètes (comptages, calendrier) : imputation, standardisation.
# Pas d'écrêtage : pour ces comptages, le 99e centile vaut souvent 0 ou 1 et
# l'écrêtage effacerait des profils rares mais informatifs.
colonnes_discretes = [
    "days_in_waiting_list",            # jours en liste d'attente (99 % de zéros)
    "arrival_date_week_number",        # semaine de l'année
    "arrival_date_day_of_month",       # jour du mois
    "adults",
    "children",
    "babies",
    "previous_cancellations",          # annulations passées du client
    "previous_bookings_not_canceled",  # séjours passés honorés
    "booking_changes",                 # modifications de la réservation
    "required_car_parking_spaces",     # places de stationnement demandées
    "total_of_special_requests",       # demandes spéciales
    "total_guests",                    # créée : nombre total de clients
]

# Indicateurs déjà codés 0/1 : imputation seulement
colonnes_binaires = [
    "is_repeated_guest",  # client déjà venu
    "has_children",       # créée : enfant ou bébé présent
    "has_agent",          # créée : réservation via une agence
    "has_company",        # créée : séjour payé par une entreprise
]

# Variables catégorielles : imputation, regroupement des modalités rares, encodage
colonnes_cat = [
    "hotel",                 # City Hotel ou Resort Hotel
    "arrival_date_month",
    "meal",                  # formule de repas
    "country",               # pays d'origine (177 modalités)
    "market_segment",        # segment de marché
    "distribution_channel",  # canal de distribution
    "reserved_room_type",    # type de chambre réservée
    "deposit_type",          # type de dépôt
    "customer_type",         # type de client
    "arrival_season",        # créée : saison d'arrivée
]

# ===========================================================================
# 1. Préparation de pipelines spécifiques (un par type de colonnes)
# ===========================================================================

# Continues : les valeurs extrêmes sont ramenées aux 1er et 99e centiles
# AVANT la standardisation, pour qu'elles ne faussent pas moyenne et écart-type.
pipeline_continu = Pipeline([
    ("imputation", SimpleImputer(strategy="median")),   # médiane : robuste à l'asymétrie
    ("ecretage", OutlierCapper(lower_quantile=0.01, upper_quantile=0.99)),  # transformateur personnalisé
    ("scaling", StandardScaler()),                      # moyenne 0, écart-type 1
])

# Discrètes : imputation puis standardisation
pipeline_discret = Pipeline([
    ("imputation", SimpleImputer(strategy="median")),
    ("scaling", StandardScaler()),
])

# Binaires : la valeur la plus fréquente remplace un éventuel manquant
pipeline_binaire = Pipeline([
    ("imputation", SimpleImputer(strategy="most_frequent")),
])

# Catégorielles
pipeline_cat = Pipeline([
    # Valeur manquante -> modalité explicite « Inconnu » (l'absence de pays est une information)
    ("imputation", SimpleImputer(strategy="constant", fill_value="Inconnu")),
    # Modalités représentant moins de 0,5 % des réservations -> « Autre » (transformateur personnalisé)
    ("regroupement_rares", RareCategoryGrouper(min_freq=0.005)),
    # Une colonne 0/1 par modalité. handle_unknown="ignore" : une modalité jamais vue
    # donne une ligne de zéros au lieu d'une erreur. sparse_output=False : tableau dense.
    ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# ===========================================================================
# 2. Combine les traitements dans un seul objet (ColumnTransformer)
# ===========================================================================
preprocessor = ColumnTransformer([
    ("num_continu", pipeline_continu, colonnes_continues),
    ("num_discret", pipeline_discret, colonnes_discretes),
    ("binaire", pipeline_binaire, colonnes_binaires),
    ("cat", pipeline_cat, colonnes_cat),
], remainder="drop")  # toute colonne non listée est écartée

# ===========================================================================
# 3. Pipeline de prétraitement complet
#    Le transformateur personnalisé HotelFeatureEngineer crée d'abord les
#    nouvelles variables ; le ColumnTransformer les route ensuite vers le bon
#    sous-pipeline. L'ordre des étapes est donc essentiel.
# ===========================================================================
pipeline_pretraitement = Pipeline([
    ("creation_variables", HotelFeatureEngineer()),  # Partie 3 : transformateur personnalisé
    ("colonnes", preprocessor),                      # Partie 2 : ColumnTransformer
])

# ===========================================================================
# 4. Sélection des caractéristiques (Partie 4)
#    Une forêt aléatoire mesure l'importance de chaque colonne prétraitée ;
#    seules les colonnes dont l'importance dépasse la médiane sont gardées.
# ===========================================================================
selecteur = SelectFromModel(
    RandomForestClassifier(
        n_estimators=100,         # forêt légère : elle sert à mesurer, pas à prédire
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",  # annulations et maintiens pèsent autant dans l'importance
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    threshold="median",
)

# ===========================================================================
# 5. Pipeline complet : prétraitement + sélection + modèle (Partie 5)
#    Le modèle indiqué ici est celui retenu à l'issue de la comparaison du
#    notebook (HistGradientBoosting avec pondération des classes). Le notebook
#    remplace l'étape "model" par d'autres algorithmes pour les comparer :
#        clone(model_pipeline).set_params(model=RandomForestClassifier(...))
# ===========================================================================
model_pipeline = Pipeline([
    ("preprocessing", pipeline_pretraitement),  # Parties 2 et 3
    ("selection", selecteur),                   # Partie 4
    ("model", HistGradientBoostingClassifier(   # Partie 5
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )),
])
