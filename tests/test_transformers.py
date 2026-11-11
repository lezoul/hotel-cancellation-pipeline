"""Tests unitaires : transformateurs personnalisés, pipelines et module prescriptif.

Exécution depuis la racine du projet : python -m pytest -q
"""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn import set_config
from sklearn.base import clone

from src import prescriptive as pr
from src.pipelines import model_pipeline, pipeline_pretraitement
from src.transformers import HotelFeatureEngineer, OutlierCapper, RareCategoryGrouper

set_config(transform_output="pandas")  # comme dans le notebook


@pytest.fixture
def reservations():
    """Deux réservations brutes couvrant les cas particuliers (Undefined, enfant manquant...)."""
    return pd.DataFrame({
        "hotel": ["City Hotel", "Resort Hotel"],
        "lead_time": [10, 300],
        "arrival_date_month": ["July", "January"],
        "arrival_date_week_number": [27, 2],
        "arrival_date_day_of_month": [1, 15],
        "stays_in_weekend_nights": [1, 2],
        "stays_in_week_nights": [2, 5],
        "adults": [2, 0],
        "children": [np.nan, 1],
        "babies": [0, 1],
        "meal": ["Undefined", "BB"],
        "country": ["PRT", "FRA"],
        "market_segment": ["Undefined", "Online TA"],
        "distribution_channel": ["TA/TO", "Undefined"],
        "is_repeated_guest": [0, 1],
        "previous_cancellations": [0, 1],
        "previous_bookings_not_canceled": [0, 2],
        "reserved_room_type": ["A", "D"],
        "booking_changes": [0, 1],
        "deposit_type": ["No Deposit", "Non Refund"],
        "agent": [9.0, np.nan],
        "company": [np.nan, 40.0],
        "days_in_waiting_list": [0, 0],
        "customer_type": ["Transient", "Contract"],
        "adr": [100.0, 80.0],
        "required_car_parking_spaces": [0, 1],
        "total_of_special_requests": [1, 0],
    })


# --- Transformateurs -------------------------------------------------------
def test_feature_engineer_cree_les_variables(reservations):
    sortie = HotelFeatureEngineer().fit_transform(reservations)
    assert sortie["total_nights"].tolist() == [3, 7]
    assert sortie["total_guests"].tolist() == [2, 2]       # enfant manquant compté 0
    assert sortie["has_children"].tolist() == [0, 1]
    assert sortie["has_agent"].tolist() == [1, 0]
    assert sortie["has_company"].tolist() == [0, 1]
    assert sortie["arrival_season"].tolist() == ["Ete", "Hiver"]
    assert sortie.loc[0, "meal"] == "SC"
    assert pd.isna(sortie.loc[0, "market_segment"])
    assert "agent" not in sortie and "company" not in sortie


def test_feature_engineer_noms_de_sortie(reservations):
    fe = HotelFeatureEngineer().fit(reservations)
    assert sorted(fe.get_feature_names_out()) == sorted(fe.transform(reservations).columns)


def test_feature_engineer_ne_modifie_pas_l_entree(reservations):
    avant = reservations.copy()
    HotelFeatureEngineer().fit_transform(reservations)
    pd.testing.assert_frame_equal(reservations, avant)


def test_regroupement_appris_sur_fit_seulement():
    train = pd.DataFrame({"country": ["PRT"] * 8 + ["FRA", "ESP"]})
    grouper = RareCategoryGrouper(min_freq=0.15).fit(train)
    sortie = grouper.transform(pd.DataFrame({"country": ["PRT", "FRA", "JPN"]}))
    assert sortie["country"].tolist() == ["PRT", "Autre", "Autre"]   # JPN jamais vu -> Autre
    assert np.asarray(grouper.get_feature_names_out()).tolist() == ["country"]      # fourni par OneToOneFeatureMixin


def test_ecretage_aux_centiles_appris():
    train = pd.DataFrame({"lead_time": np.arange(101, dtype=float)})
    capper = OutlierCapper(0.05, 0.95).fit(train)
    sortie = capper.transform(pd.DataFrame({"lead_time": [-10.0, 50.0, 1000.0]}))
    assert sortie["lead_time"].tolist() == [5.0, 50.0, 95.0]


def test_transformateurs_clonables():
    for t in (HotelFeatureEngineer(supprimer_identifiants=False), RareCategoryGrouper(0.1), OutlierCapper(0.02, 0.98)):
        assert cast(Any, clone(t)).get_params() == t.get_params()


# --- Pipelines --------------------------------------------------------------
def test_pipeline_pretraitement_sans_valeur_manquante(reservations):
    sortie = cast(Any, clone(pipeline_pretraitement)).fit_transform(reservations)
    assert sortie.isna().sum().sum() == 0
    assert len(sortie) == len(reservations)  # un pipeline ne supprime jamais de lignes


def test_model_pipeline_trois_etapes():
    assert [nom for nom, _ in model_pipeline.steps] == ["preprocessing", "selection", "model"]


# --- Prescriptif ------------------------------------------------------------
def test_tranches_de_delai():
    tranches = pr.delay_category([0, 6, 7, 29, 30, 59, 60, 89, 90, 400]).tolist()
    assert tranches == [pr.DELAY_LABELS[i] for i in (0, 0, 1, 1, 2, 2, 3, 3, 4, 4)]


def test_loi_poisson_binomiale():
    np.testing.assert_allclose(pr.show_distribution([0.5, 0.5]), [0.25, 0.5, 0.25])
    assert np.isclose(pr.show_distribution(np.random.rand(30)).sum(), 1.0)


def test_surreservation_nulle_sans_annulation():
    plan = pr.overbooking_plan([0.0] * 10, capacity=10, p_cancel_new=0.0, max_extra=5)
    assert pr.max_safe_overbooking(plan, 0.05)["reservations_supplementaires"] == 0


def test_acompte_remboursable_rembourse_les_annulations_precoces():
    reservations = pd.DataFrame({
        "is_canceled": [1, 1, 0], "adr": [100.0, 100.0, 100.0], "nights": [2, 2, 2],
        "stay_revenue": [200.0, 200.0, 200.0], "cancel_delay": [5, 60, 0],
        "delay_category": pr.delay_category([5, 60, 0]).values,
    })
    revente = pd.Series(0.5, index=pr.DELAY_LABELS)
    bilan = pr.deposit_scenarios(reservations, [True, True, True], revente,
                                 {"test": ("part", 0.5, 30)}).iloc[0]
    assert bilan["acompte_conserve"] == 100.0       # seule l'annulation tardive (5 jours) est conservée
    assert bilan["fideles_exposes"] == 1
    assert bilan["taux_fuite_max_acceptable"] == 0.5  # 100 € conservés / 200 € de revenu fidèle


def test_reservation_non_remboursable_aucune_perte():
    # Séjour payé d'avance : l'hôtel garde le paiement, la perte est nulle même à 100 % de risque
    reco = pr.recommend_actions([1.0, 1.0, 0.9], [550.0, 550.0, 200.0], low=0.17, high=0.54,
                                deposit_type=["Non Refund", "No Deposit", "Refundable"])
    assert reco["perte_esperee"].tolist() == [0.0, 550.0, 180.0]
    assert reco["action_recommandee"].tolist() == [
        "Séjour déjà payé", "Acompte et surréservation", "Surréservation (dépôt déjà versé)"]


def test_acompte_base_commune_et_reservations_deja_payees_exclues():
    reservations = pd.DataFrame({
        "is_canceled": [1, 1, 1, 0], "adr": [100.0] * 4, "nights": [2] * 4, "stay_revenue": [200.0] * 4,
        "cancel_delay": [5, 5, 5, 0], "delay_category": pr.delay_category([5, 5, 5, 0]).values,
        "deposit_type": ["No Deposit", "No Deposit", "Non Refund", "No Deposit"],
    })
    revente = pd.Series(0.0, index=pr.DELAY_LABELS)
    politique = {"test": ("part", 0.5, None)}
    etroit = pr.deposit_scenarios(reservations, [True, False, True, False], revente, politique).iloc[0]
    large = pr.deposit_scenarios(reservations, [True, True, True, True], revente, politique).iloc[0]
    # Même perte de départ pour les deux seuils : les 2 annulations sans dépôt (2 x 200 €)
    assert etroit["perte_nette_totale"] == large["perte_nette_totale"] == 400.0
    # La réservation non remboursable (déjà payée) ne reçoit jamais d'acompte
    assert etroit["reservations_signalees"] == 1 and large["reservations_signalees"] == 3
    # Signaler plus large récupère plus et laisse une perte restante plus faible
    assert large["acompte_conserve"] > etroit["acompte_conserve"]
    assert large["perte_nette_restante"] < etroit["perte_nette_restante"]


def test_point_de_bascule_entre_deux_seuils():
    large = pd.Series({"acompte_conserve": 400.0, "revenu_fideles_exposes": 2000.0})
    etroit = pd.Series({"acompte_conserve": 300.0, "revenu_fideles_exposes": 500.0})
    bascule = pr.switch_leak_rate(large, etroit)
    assert np.isclose(bascule, 100.0 / 1500.0)
    # Au point de bascule, les deux seuils rapportent le même gain net
    assert np.isclose(pr.net_gain_by_leak_rate(large, [bascule])[0], pr.net_gain_by_leak_rate(etroit, [bascule])[0])
