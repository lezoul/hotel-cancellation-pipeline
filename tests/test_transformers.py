"""Tests unitaires des transformateurs personnalisés et du module prescriptif.

Exécution : python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.prescriptive import overbooking_plan, recommend_actions, show_distribution
from src.transformers import HotelFeatureEngineer, OutlierCapper, RareCategoryGrouper


@pytest.fixture
def raw_rows():
    return pd.DataFrame(
        {
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
        }
    )


def test_feature_engineer_creates_expected_columns(raw_rows):
    out = HotelFeatureEngineer().fit_transform(raw_rows)
    assert out["total_nights"].tolist() == [3, 7]
    assert out["total_guests"].tolist() == [2, 2]          # enfant manquant compté 0
    assert out["has_children"].tolist() == [0, 1]
    assert out["has_agent"].tolist() == [1, 0]
    assert out["has_company"].tolist() == [0, 1]
    assert out["arrival_season"].tolist() == ["Ete", "Hiver"]
    assert out.loc[0, "meal"] == "SC"
    assert pd.isna(out.loc[0, "market_segment"])
    assert "agent" not in out and "company" not in out
    assert list(out.columns) == list(HotelFeatureEngineer().fit(raw_rows).get_feature_names_out())


def test_feature_engineer_does_not_modify_input(raw_rows):
    before = raw_rows.copy()
    HotelFeatureEngineer().fit_transform(raw_rows)
    pd.testing.assert_frame_equal(raw_rows, before)


def test_rare_category_grouper_learns_on_fit_only():
    train = pd.DataFrame({"country": ["PRT"] * 8 + ["FRA", "ESP"]})
    grouper = RareCategoryGrouper(min_freq=0.15).fit(train)
    out = grouper.transform(pd.DataFrame({"country": ["PRT", "FRA", "JPN"]}))
    assert out["country"].tolist() == ["PRT", "Autre", "Autre"]


def test_outlier_capper_clips_to_train_quantiles():
    train = pd.DataFrame({"lead_time": np.arange(101, dtype=float)})
    capper = OutlierCapper(0.05, 0.95).fit(train)
    out = capper.transform(pd.DataFrame({"lead_time": [-10.0, 50.0, 1000.0]}))
    assert out["lead_time"].tolist() == [5.0, 50.0, 95.0]


def test_outlier_capper_rejects_bad_quantiles():
    with pytest.raises(ValueError):
        OutlierCapper(0.9, 0.1).fit(pd.DataFrame({"a": [1.0, 2.0]}))


def test_transformers_are_clonable():
    for t in (HotelFeatureEngineer(drop_ids=False), RareCategoryGrouper(0.1), OutlierCapper(0.02, 0.98)):
        assert clone(t).get_params() == t.get_params()


def test_show_distribution_sums_to_one_and_matches_binomial():
    dist = show_distribution([0.5, 0.5])
    np.testing.assert_allclose(dist, [0.25, 0.5, 0.25])
    assert np.isclose(show_distribution(np.random.rand(30)).sum(), 1.0)


def test_overbooking_zero_cancellation_means_no_extra_booking():
    plan = overbooking_plan([0.0] * 10, capacity=10, p_cancel_new=0.0, room_revenue=100, walk_cost=300, max_extra=5)
    assert plan.loc[plan["profit_espere"].idxmax(), "reservations_supplementaires"] == 0


def test_recommend_actions_prefers_no_action_for_low_risk():
    rec = recommend_actions([0.01, 0.9], [200.0, 600.0])
    assert rec.loc[0, "action_recommandee"] == "Aucune action"
    assert rec.loc[1, "action_recommandee"] != "Aucune action"
