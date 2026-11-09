"""Configuration centrale du projet.

Constantes partagées par le notebook, les modules de src/ et l'application
Streamlit : chemins des fichiers, colonnes de fuite, paramètres de
reproductibilité. Les listes de colonnes de chaque pipeline sont écrites
directement dans le notebook, à côté des pipelines qui les utilisent.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]           # racine du projet
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "hotels.csv"           # jeu de données brut
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"            # graphiques exportés pour les diapositives

MODEL_PATH = MODELS_DIR / "pipeline_annulation.joblib"    # pipeline complet entraîné
METADATA_PATH = MODELS_DIR / "metadata.json"              # seuils, métriques, date d'entraînement
TEST_SET_PATH = MODELS_DIR / "test_set.csv"               # ensemble de test (pour l'application)

# Source officielle du jeu de données (TidyTuesday, 2020-02-11)
DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/"
    "data/2020/2020-02-11/hotels.csv"
)

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
RANDOM_STATE = 42   # même graine pour tous les tirages aléatoires
TEST_SIZE = 0.20    # 20 % des réservations réservées au test final

# ---------------------------------------------------------------------------
# Variable cible
# ---------------------------------------------------------------------------
TARGET = "is_canceled"  # 1 = annulée, 0 = maintenue

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

# ---------------------------------------------------------------------------
# Colonnes brutes en entrée du pipeline complet
# ---------------------------------------------------------------------------
# arrival_date_year n'y figure pas : 2015 à 2017 ne se reproduiront pas, une
# nouvelle année n'existerait pas dans l'apprentissage.
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

# Mois dans l'ordre du calendrier (formulaire de l'application)
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
