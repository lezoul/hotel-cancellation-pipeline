"""Configuration centrale du projet.

Toutes les constantes (chemins, colonnes, paramètres de reproductibilité)
sont regroupées ici pour qu'un seul fichier serve de référence aux scripts,
au notebook et à l'application Streamlit.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "hotels.csv"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

MODEL_PATH = MODELS_DIR / "pipeline_annulation.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
TEST_SET_PATH = MODELS_DIR / "test_set.csv"

# Source officielle du jeu de données (TidyTuesday, 2020-02-11)
DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/"
    "data/2020/2020-02-11/hotels.csv"
)

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Variable cible
# ---------------------------------------------------------------------------
TARGET = "is_canceled"

# ---------------------------------------------------------------------------
# Colonnes retirées avant la modélisation
# ---------------------------------------------------------------------------
# Fuite de données : ces informations ne sont connues qu'APRÈS l'issue de la
# réservation (annulation ou séjour). Les garder donnerait un modèle
# artificiellement parfait et inutilisable au moment de la prédiction.
LEAKAGE_COLUMNS = [
    "reservation_status",       # Canceled / Check-Out / No-Show : quasi la cible
    "reservation_status_date",  # date du dernier statut, connue après coup
    "assigned_room_type",       # chambre attribuée à l'arrivée du client
]

# Colonnes non pertinentes pour généraliser à de futures réservations.
NON_GENERALIZABLE_COLUMNS = [
    "arrival_date_year",  # 2015 à 2017 : une nouvelle année n'existe pas dans l'apprentissage
]

# ---------------------------------------------------------------------------
# Colonnes après ingénierie des caractéristiques (HotelFeatureEngineer)
# ---------------------------------------------------------------------------
# Variables continues à forte asymétrie : imputation, écrêtage (OutlierCapper), standardisation
CONTINUOUS_FEATURES = [
    "lead_time",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    # créées par HotelFeatureEngineer
    "total_nights",
    "adr_per_person",
]

# Variables discrètes (comptages, calendrier) : imputation, standardisation.
# Pas d'écrêtage : pour ces comptages, le 99e centile vaut souvent 0 ou 1,
# l'écrêtage effacerait l'information des clients atypiques (ex. 26 annulations).
# days_in_waiting_list est ici pour la même raison : plus de 99 % des valeurs
# valent 0 ; l'écrêter au 99e centile transformerait la colonne en constante.
DISCRETE_FEATURES = [
    "days_in_waiting_list",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "required_car_parking_spaces",
    "total_of_special_requests",
    # créée par HotelFeatureEngineer
    "total_guests",
]

NUMERIC_FEATURES = CONTINUOUS_FEATURES + DISCRETE_FEATURES

# Indicateurs binaires 0/1 : imputation seulement (aucune mise à l'échelle utile)
BINARY_FEATURES = [
    "is_repeated_guest",
    # créés par HotelFeatureEngineer
    "has_children",
    "has_agent",
    "has_company",
]

# Variables catégorielles : imputation, regroupement des modalités rares, encodage one-hot
CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    # créée par HotelFeatureEngineer
    "arrival_season",
]

# Colonnes brutes attendues en entrée du pipeline complet (avant ingénierie)
RAW_INPUT_FEATURES = [
    "hotel",
    "lead_time",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ---------------------------------------------------------------------------
# Paramètres des transformateurs personnalisés
# ---------------------------------------------------------------------------
RARE_CATEGORY_MIN_FREQ = 0.005   # modalité < 0,5 % des lignes -> « Autre »
CAP_LOWER_QUANTILE = 0.01        # écrêtage des valeurs extrêmes (winsorisation)
CAP_UPPER_QUANTILE = 0.99
