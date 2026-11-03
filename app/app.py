"""Application Streamlit : analyse prédictive ET prescriptive des annulations.

Lancement (depuis la racine du projet) :
    streamlit run app/app.py

Prérequis : un pipeline entraîné dans models/ (notebook exécuté ou
`python -m src.train --fast`).

Onglets
-------
1. Réservation        : probabilité d'annulation d'une réservation et action recommandée
2. Lot de réservations: priorisation d'un ensemble de réservations, export CSV
3. Seuil de décision  : seuil qui minimise le coût des erreurs selon vos hypothèses
4. Surréservation     : nombre optimal de réservations à accepter au-delà de la capacité
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.model import load_metadata, load_pipeline  # noqa: E402
from src.prescriptive import (  # noqa: E402
    Action,
    best_overbooking,
    cost_by_threshold,
    default_actions,
    overbooking_plan,
    recommend_actions,
    revenue_at_risk,
    risk_level,
)

st.set_page_config(page_title="Annulations hôtelières", page_icon="🏨", layout="wide")

BLUE, RED, GREY = "#2a6f97", "#c44536", "#8d99ae"
RISK_COLORS = {"Faible": "#2a9d8f", "Moyen": "#e9c46a", "Élevé": RED}


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return load_pipeline()


@st.cache_data
def get_metadata():
    return load_metadata()


@st.cache_data
def get_test_set():
    return pd.read_csv(config.TEST_SET_PATH)


@st.cache_data
def score(df: pd.DataFrame) -> np.ndarray:
    return get_pipeline().predict_proba(df[config.RAW_INPUT_FEATURES])[:, 1]


if not config.MODEL_PATH.exists():
    st.error(
        "Aucun modèle entraîné n'a été trouvé dans `models/`.\n\n"
        "Exécutez le notebook `notebooks/evaluation_sommative_2.ipynb` ou lancez :\n\n"
        "`python -m src.train --fast`"
    )
    st.stop()

pipeline = get_pipeline()
meta = get_metadata()
test_df = get_test_set()
test_proba = score(test_df)


# ---------------------------------------------------------------------------
# Barre latérale : hypothèses métier
# ---------------------------------------------------------------------------
st.sidebar.title("🏨 Hypothèses métier")
st.sidebar.caption(
    "Les recommandations dépendent de ces hypothèses. Elles ne sont pas estimées par le modèle : "
    "elles traduisent la réalité économique de l'hôtel et doivent être validées (test A/B)."
)
resale_rate = st.sidebar.slider(
    "Part des chambres annulées revendues", 0.0, 0.9, 0.30, 0.05,
    help="Une annulation tardive est rarement revendue. 30 % = 70 % du séjour est perdu.",
)
low_risk, high_risk = st.sidebar.slider(
    "Bornes des niveaux de risque", 0.0, 1.0, (0.30, 0.60), 0.05,
    help="Faible sous la première borne, élevé au-dessus de la seconde.",
)

st.sidebar.subheader("Catalogue d'actions")
actions_df = pd.DataFrame([vars(a) for a in default_actions()])
actions_df = st.sidebar.data_editor(
    actions_df[["nom", "cout", "efficacite", "desistement"]],
    column_config={
        "nom": st.column_config.TextColumn("Action", disabled=True),
        "cout": st.column_config.NumberColumn("Coût (€)", min_value=0.0, step=1.0),
        "efficacite": st.column_config.NumberColumn("Efficacité", min_value=0.0, max_value=1.0, step=0.05,
                                                    help="Part des annulations évitées"),
        "desistement": st.column_config.NumberColumn("Désistement", min_value=0.0, max_value=1.0, step=0.01,
                                                     help="Part des bons clients perdus à cause de l'action"),
    },
    hide_index=True,
    key="actions",
)
descriptions = {a.nom: a.description for a in default_actions()}
actions = [
    Action(r.nom, float(r.cout), float(r.efficacite), float(r.desistement), descriptions.get(r.nom, ""))
    for r in actions_df.itertuples()
]

st.sidebar.divider()
m = meta["metriques_test_seuil_optimal"]
st.sidebar.subheader("Modèle")
st.sidebar.markdown(
    f"**{meta['modele']}**  \n"
    f"ROC-AUC test : **{m['roc_auc']:.3f}**  \n"
    f"Rappel au seuil {meta['seuil_optimal']:.2f} : **{m['recall']:.1%}**  \n"
    f"Entraîné le {meta['date_entrainement'][:10]} sur {meta['n_entrainement']:,} réservations".replace(",", " ")
)


def euro(v: float) -> str:
    return f"{v:,.0f} €".replace(",", " ")


def actions_table(p: float, revenue: float) -> pd.DataFrame:
    rows = []
    for a in actions:
        gain = p * revenue * a.efficacite - a.cout - (1 - p) * revenue * a.desistement
        rows.append({"Action": a.nom, "Description": a.description, "Gain net espéré (€)": round(gain, 2)})
    return pd.DataFrame(rows).sort_values("Gain net espéré (€)", ascending=False)


# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.title("Prévoir les annulations, décider quoi faire")
st.markdown(
    "Le modèle **prédit** la probabilité qu'une réservation soit annulée. L'application la transforme "
    "en **décisions** : quelle action mener, à partir de quel risque agir, combien de chambres surréserver."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔎 Réservation", "📋 Lot de réservations", "🎚️ Seuil de décision", "🛏️ Surréservation"]
)

# ---------------------------------------------------------------------------
# 1. Réservation unique
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Évaluer une réservation")
    countries = meta["valeurs_reference"]["countries"]
    with st.form("reservation"):
        c1, c2, c3, c4 = st.columns(4)
        hotel = c1.selectbox("Hôtel", ["City Hotel", "Resort Hotel"])
        lead_time = c1.number_input("Délai avant l'arrivée (jours)", 0, 800, 120)
        month = c1.selectbox("Mois d'arrivée", config.MONTHS, index=7)
        day = c1.number_input("Jour d'arrivée", 1, 31, 15)

        weekend = c2.number_input("Nuits de fin de semaine", 0, 20, 2)
        week = c2.number_input("Nuits de semaine", 0, 50, 3)
        adults = c2.number_input("Adultes", 0, 10, 2)
        children = c2.number_input("Enfants", 0, 10, 0)
        babies = c2.number_input("Bébés", 0, 10, 0)

        adr = c3.number_input("Tarif journalier moyen (€)", 0.0, 1000.0, 110.0, step=5.0)
        deposit = c3.selectbox("Type de dépôt", ["No Deposit", "Non Refund", "Refundable"])
        market = c3.selectbox(
            "Segment de marché",
            ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary", "Aviation"],
        )
        channel = c3.selectbox("Canal de distribution", ["TA/TO", "Direct", "Corporate", "GDS"])
        customer = c3.selectbox("Type de client", ["Transient", "Transient-Party", "Contract", "Group"])

        country = c4.selectbox("Pays", countries, index=countries.index("PRT") if "PRT" in countries else 0)
        meal = c4.selectbox("Repas", ["BB", "HB", "FB", "SC"])
        room = c4.selectbox("Type de chambre réservée", list("ABCDEFGHL"))
        special = c4.number_input("Demandes spéciales", 0, 5, 0)
        parking = c4.number_input("Places de stationnement", 0, 3, 0)

        with st.expander("Historique du client et autres informations"):
            e1, e2, e3 = st.columns(3)
            repeated = e1.checkbox("Client déjà venu")
            prev_cancel = e1.number_input("Annulations passées", 0, 30, 0)
            prev_ok = e2.number_input("Séjours passés non annulés", 0, 80, 0)
            changes = e2.number_input("Modifications de la réservation", 0, 20, 0)
            waiting = e3.number_input("Jours en liste d'attente", 0, 400, 0)
            via_agent = e3.checkbox("Réservé via une agence", value=True)
            via_company = e3.checkbox("Payé par une entreprise")
        submitted = st.form_submit_button("Évaluer", type="primary")

    week_number = int(pd.Timestamp(2017, config.MONTHS.index(month) + 1, min(int(day), 28)).isocalendar().week)
    booking = pd.DataFrame([{
        "hotel": hotel, "lead_time": lead_time, "arrival_date_month": month,
        "arrival_date_week_number": week_number, "arrival_date_day_of_month": day,
        "stays_in_weekend_nights": weekend, "stays_in_week_nights": week, "adults": adults,
        "children": children, "babies": babies, "meal": meal, "country": country,
        "market_segment": market, "distribution_channel": channel, "is_repeated_guest": int(repeated),
        "previous_cancellations": prev_cancel, "previous_bookings_not_canceled": prev_ok,
        "reserved_room_type": room, "booking_changes": changes, "deposit_type": deposit,
        "agent": 9.0 if via_agent else np.nan, "company": 40.0 if via_company else np.nan,
        "days_in_waiting_list": waiting, "customer_type": customer, "adr": adr,
        "required_car_parking_spaces": parking, "total_of_special_requests": special,
    }])

    p = float(pipeline.predict_proba(booking)[0, 1])
    revenue = float(revenue_at_risk([adr], [weekend + week], resale_rate)[0])
    level = risk_level(p, low_risk, high_risk)
    table = actions_table(p, revenue)
    best = table.iloc[0]

    g1, g2 = st.columns([1, 2])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": " %", "valueformat": ".1f"},
            title={"text": f"Risque {level.lower()}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS[level]},
                "steps": [
                    {"range": [0, low_risk * 100], "color": "#e8f4f2"},
                    {"range": [low_risk * 100, high_risk * 100], "color": "#fbf3dc"},
                    {"range": [high_risk * 100, 100], "color": "#f8e1de"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": meta["seuil_optimal"] * 100},
            },
        ))
        gauge.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(gauge, width="stretch")
        st.caption(f"Trait noir : seuil de décision du modèle ({meta['seuil_optimal']:.2f}).")
    with g2:
        k1, k2, k3 = st.columns(3)
        k1.metric("Probabilité d'annulation", f"{p:.1%}")
        k2.metric("Revenu en jeu", euro(revenue), help="Tarif x nuits x (1 - part revendue)")
        k3.metric("Perte espérée", euro(p * revenue))
        st.success(f"**Action recommandée : {best['Action']}**  \n{best['Description']}")
        st.dataframe(table, hide_index=True, width="stretch")

    st.markdown("##### Sensibilité au délai de réservation")
    st.caption(
        "Comment la probabilité prédite évolue si seul le délai change. Il s'agit de la réponse du modèle, "
        "pas d'un effet causal garanti."
    )
    grid = np.arange(0, 401, 10)
    sens = pd.concat([booking] * len(grid), ignore_index=True).assign(lead_time=grid)
    sens_p = pipeline.predict_proba(sens)[:, 1]
    fig = px.line(x=grid, y=sens_p, labels={"x": "Délai avant l'arrivée (jours)", "y": "Probabilité d'annulation"})
    fig.add_vline(x=lead_time, line_dash="dash", line_color=RED)
    fig.update_traces(line_color=BLUE)
    fig.update_layout(height=300, yaxis_tickformat=".0%", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 2. Lot de réservations
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Prioriser un ensemble de réservations")
    source = st.radio(
        "Source", ["Échantillon de l'ensemble de test", "Importer un fichier CSV"], horizontal=True
    )
    # Pas de st.stop() ici : il interromprait aussi le rendu des onglets suivants
    batch = None
    if source == "Importer un fichier CSV":
        st.caption("Le fichier doit contenir les colonnes brutes du jeu hotels.csv (les colonnes en trop sont ignorées).")
        upload = st.file_uploader("Fichier CSV", type="csv")
        if upload is None:
            st.info("Importez un fichier pour lancer l'analyse.")
        else:
            uploaded = pd.read_csv(upload, na_values=["NULL", ""])
            missing = sorted(set(config.RAW_INPUT_FEATURES) - set(uploaded.columns))
            if missing:
                st.error(f"Colonnes manquantes : {missing}")
            else:
                batch = uploaded.reset_index(drop=True)
    else:
        n = st.slider("Nombre de réservations", 50, 2000, 500, 50)
        batch = test_df.sample(n, random_state=config.RANDOM_STATE).reset_index(drop=True)

    if batch is not None:
        batch_p = score(batch)
        nights = batch["stays_in_weekend_nights"] + batch["stays_in_week_nights"]
        rev = revenue_at_risk(batch["adr"], nights, resale_rate)
        reco = recommend_actions(batch_p, rev, actions)
        reco["niveau_risque"] = [risk_level(v, low_risk, high_risk) for v in batch_p]
        out = pd.concat(
            [batch[["hotel", "arrival_date_month", "arrival_date_day_of_month", "lead_time", "adr",
                    "market_segment", "deposit_type", "country"]], reco], axis=1
        ).sort_values("perte_esperee", ascending=False)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Réservations", f"{len(out):,}".replace(",", " "))
        k2.metric("Annulations attendues", f"{batch_p.sum():.0f}", f"{batch_p.mean():.1%} du lot", delta_color="off")
        k3.metric("Perte espérée totale", euro(reco["perte_esperee"].sum()))
        k4.metric("Gain net espéré des actions", euro(reco["gain_net_espere"].sum()))

        c1, c2 = st.columns(2)
        hist = px.histogram(out, x="proba_annulation", color="niveau_risque", nbins=40,
                            color_discrete_map=RISK_COLORS, labels={"proba_annulation": "Probabilité d'annulation"})
        hist.update_layout(height=320, bargap=0.05, legend_title_text="Risque", margin=dict(t=30, b=10))
        c1.plotly_chart(hist, width="stretch")
        agg = out.groupby("action_recommandee").agg(reservations=("proba_annulation", "size"),
                                                      gain=("gain_net_espere", "sum")).reset_index()
        bar = px.bar(agg, x="reservations", y="action_recommandee", orientation="h", text="reservations",
                     labels={"reservations": "Réservations", "action_recommandee": ""}, color_discrete_sequence=[BLUE])
        bar.update_layout(height=320, margin=dict(t=30, b=10))
        c2.plotly_chart(bar, width="stretch")

        st.markdown("##### Réservations à traiter en priorité (perte espérée décroissante)")
        st.dataframe(
            out,
            hide_index=True,
            width="stretch",
            column_config={
                "proba_annulation": st.column_config.ProgressColumn("Probabilité", format="percent", min_value=0, max_value=1),
                "revenu_en_jeu": st.column_config.NumberColumn("Revenu en jeu (€)", format="%.0f"),
                "perte_esperee": st.column_config.NumberColumn("Perte espérée (€)", format="%.0f"),
                "gain_net_espere": st.column_config.NumberColumn("Gain net (€)", format="%.1f"),
            },
        )
        st.download_button("Télécharger les recommandations (CSV)", out.to_csv(index=False).encode("utf-8"),
                           "recommandations_annulations.csv", "text/csv")

# ---------------------------------------------------------------------------
# 3. Seuil de décision
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("À partir de quel risque faut-il agir ?")
    st.markdown(
        "Un **faux négatif** est une annulation que l'on n'a pas vue venir ; un **faux positif** est une action "
        "menée auprès d'un client qui serait venu. Le seuil optimal minimise le coût total des deux erreurs."
    )
    nights_t = test_df["stays_in_weekend_nights"] + test_df["stays_in_week_nights"]
    default_fn = float(np.mean(revenue_at_risk(test_df["adr"], nights_t, resale_rate)) * 0.35)
    c1, c2 = st.columns(2)
    cost_fn = c1.number_input("Coût d'un faux négatif (€)", 0.0, 2000.0, round(default_fn, 0), 5.0,
                              help="Par défaut : revenu moyen en jeu x 35 % (part des annulations qu'une action aurait évitée)")
    cost_fp = c2.number_input("Coût d'un faux positif (€)", 0.0, 500.0, 30.0, 5.0,
                              help="Coût de l'action et perte de bienveillance du client")

    y_test = test_df[config.TARGET].values
    curve = cost_by_threshold(y_test, test_proba, cost_fn, cost_fp)
    best_row = curve.loc[curve["cout_total"].idxmin()]
    at_05 = curve.loc[(curve["seuil"] - 0.5).abs().idxmin()]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Seuil optimal", f"{best_row['seuil']:.2f}")
    k2.metric("Coût au seuil optimal", euro(best_row["cout_total"]))
    k3.metric("Économie vs seuil 0,50", euro(at_05["cout_total"] - best_row["cout_total"]))
    k4.metric("Annulations détectées (rappel)", f"{best_row['rappel']:.1%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["seuil"], y=curve["cout_total"], name="Coût total", line_color=BLUE))
    fig.add_vline(x=best_row["seuil"], line_dash="dash", line_color=RED, annotation_text="optimum")
    fig.add_vline(x=0.5, line_dash="dot", line_color=GREY, annotation_text="0,50")
    fig.update_layout(height=350, xaxis_title="Seuil de décision", yaxis_title="Coût total (€)", margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    pr = go.Figure()
    pr.add_trace(go.Scatter(x=curve["seuil"], y=curve["rappel"], name="Rappel (annulations détectées)", line_color=RED))
    pr.add_trace(go.Scatter(x=curve["seuil"], y=curve["precision"], name="Précision (alertes justifiées)", line_color=BLUE))
    pr.add_vline(x=best_row["seuil"], line_dash="dash", line_color=RED)
    pr.update_layout(height=300, xaxis_title="Seuil de décision", yaxis_tickformat=".0%", margin=dict(t=30),
                     legend=dict(orientation="h", y=1.15))
    st.plotly_chart(pr, width="stretch")
    st.caption(
        f"Calculé sur l'ensemble de test ({len(test_df):,} réservations) pour explorer vos hypothèses. "
        "Dans le notebook, le seuil de référence est choisi sur des prédictions hors pli de l'entraînement."
        .replace(",", " ")
    )

# ---------------------------------------------------------------------------
# 4. Surréservation
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Combien de réservations accepter au-delà de la capacité ?")
    st.markdown(
        "Pour une date d'arrivée, chaque réservation a sa propre probabilité d'annulation. Le nombre de clients "
        "présents suit une **loi de Poisson binomiale**, calculée exactement. On cherche le nombre de réservations "
        "supplémentaires qui maximise : *revenu x chambres occupées - pénalité x clients à reloger*."
    )
    meta_df = test_df.assign(proba=test_proba)
    c1, c2 = st.columns(2)
    hotel_sel = c1.selectbox("Hôtel ", ["City Hotel", "Resort Hotel"], key="ob_hotel")
    sub = meta_df[meta_df["hotel"] == hotel_sel]
    counts = (
        sub.groupby(["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).size()
        .sort_values(ascending=False).head(30)
    )
    labels = [f"{d} {mo} {y} ({n} réservations)" for (y, mo, d), n in counts.items()]
    choice = c2.selectbox("Date d'arrivée (30 dates les plus chargées du test)", range(len(labels)),
                          format_func=lambda i: labels[i])
    y_, mo_, d_ = counts.index[choice]
    carnet = sub[(sub["arrival_date_year"] == y_) & (sub["arrival_date_month"] == mo_)
                 & (sub["arrival_date_day_of_month"] == d_)]

    stay_rev = carnet["adr"] * (carnet["stays_in_weekend_nights"] + carnet["stays_in_week_nights"]).clip(lower=1)
    c1, c2, c3, c4 = st.columns(4)
    capacity = c1.number_input("Capacité (chambres)", 1, 500, len(carnet),
                               help="Par défaut : hôtel complet, une chambre par réservation au carnet")
    room_revenue = c2.number_input("Revenu d'une chambre occupée (€)", 10.0, 5000.0,
                                   float(round(stay_rev.median(), 0)), 10.0)
    walk_mult = c3.slider("Pénalité de relogement (x revenu)", 1.0, 6.0, 2.5, 0.5,
                          help="Relogement dans un autre hôtel, dédommagement, atteinte à la réputation")
    p_new = c4.slider("Probabilité d'annulation des nouvelles réservations", 0.0, 0.9,
                      float(round(carnet["proba"].mean(), 2)), 0.01)

    plan = overbooking_plan(carnet["proba"], int(capacity), p_new, room_revenue, walk_mult * room_revenue,
                            max_extra=max(int(capacity), 10))
    best = best_overbooking(plan)
    base = plan.iloc[0]
    k = int(best["reservations_supplementaires"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Réservations supplémentaires", f"+{k}")
    k2.metric("Taux d'occupation espéré", f"{best['taux_occupation_espere']:.1%}",
              f"{(best['taux_occupation_espere'] - base['taux_occupation_espere']) * 100:+.1f} pts")
    k3.metric("Gain de profit espéré", euro(best["profit_espere"] - base["profit_espere"]))
    k4.metric("Risque d'au moins un relogement", f"{best['prob_au_moins_un_reloge']:.1%}",
              f"{best['clients_reloges_esperes']:.2f} client(s) en moyenne", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plan["reservations_supplementaires"], y=plan["profit_espere"],
                             mode="lines+markers", line_color=BLUE, name="Profit espéré"))
    fig.add_vline(x=k, line_dash="dash", line_color=RED, annotation_text=f"optimum +{k}")
    fig.update_layout(height=350, xaxis_title="Réservations acceptées au-delà de la capacité",
                      yaxis_title="Profit espéré (€)", margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Vérification a posteriori sur cette date")
    v1, v2, v3 = st.columns(3)
    v1.metric("Réservations au carnet", len(carnet))
    v2.metric("Annulations prévues (espérance)", f"{carnet['proba'].sum():.1f}")
    v3.metric("Annulations réellement observées", int(carnet[config.TARGET].sum()))
    st.caption(
        "Les dates proviennent de l'ensemble de test (20 % des réservations), donc un carnet réduit. "
        "Le principe est identique sur un carnet complet."
    )
