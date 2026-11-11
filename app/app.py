"""Application Streamlit : prédiction des annulations et analyse prescriptive.

Lancement (depuis la racine du projet) :
    streamlit run app/app.py

Prérequis : avoir exécuté notebooks/01_pipeline.ipynb, qui enregistre le
pipeline entraîné dans models/.

Onglets
-------
1. Réservation         : probabilité d'annulation, zone de risque et action
2. Lot de réservations : priorisation d'un ensemble de réservations, export CSV
3. Pertes et acomptes  : moment des annulations, revente, manque à gagner,
                         simulateur d'acompte avec seuil de rentabilité
4. Surréservation      : réservations supplémentaires selon le risque de relogement accepté

Toute la logique de calcul est dans src/prescriptive.py (aucun chiffre inventé :
seuls le niveau d'acompte et le niveau de service sont des choix de l'utilisateur).
"""

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Racine du projet ajoutée au chemin pour importer src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import prescriptive as pr  # noqa: E402
from src.sauvegarde import load_metadata, load_pipeline  # noqa: E402

st.set_page_config(page_title="Annulations hôtelières", page_icon="🏨", layout="wide")

BLEU, ROUGE, GRIS, VERT = "#2a6f97", "#c44536", "#8d99ae", "#2a9d8f"
COULEURS_RISQUE = {"Faible": VERT, "Modéré": "#e9c46a", "Élevé": ROUGE}


# ---------------------------------------------------------------------------
# Chargement (mis en cache : calculé une seule fois par session)
# ---------------------------------------------------------------------------
@st.cache_resource
def charger_modele():
    return load_pipeline()


@st.cache_data
def charger_metadata():
    return load_metadata()


@st.cache_data
def charger_test():
    """Ensemble de test avec les dates réelles d'annulation (même séparation que le notebook 01)."""
    df = pr.add_dates(dl.clean_data(dl.load_raw_data(), verbose=False, drop_leakage=False))
    X, y = dl.split_features_target(df)
    _, X_test, _, _ = dl.train_test_data(X, y)
    test = df.loc[X_test.index].copy()
    test["proba"] = charger_modele().predict_proba(X_test)[:, 1]
    return df, test


@st.cache_data
def charger_toutes():
    """Toutes les réservations, doublons compris (bilan financier : chaque chambre compte)."""
    return pr.load_all_bookings()


@st.cache_data
def charger_revente():
    """Taux de revente par tranche de délai, estimé à partir de l'occupation réelle."""
    annulations, capacite, _ = pr.estimate_resale()
    return pr.resale_by_delay(annulations), capacite


def euro(v: float) -> str:
    return f"{v:,.0f} €".replace(",", " ")


if not config.MODEL_PATH.exists():
    st.error("Aucun modèle entraîné dans `models/`. Exécutez d'abord `notebooks/01_pipeline.ipynb`.")
    st.stop()

modele = charger_modele()
meta = charger_metadata()
SEUIL_F2, SEUIL_F05 = float(meta["seuil_f2"]), float(meta["seuil_f05"])

# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------
st.sidebar.title("🏨 Modèle")
m = meta["metriques_test_seuil_f2"]
st.sidebar.markdown(
    f"**{meta['modele']}** (calibré)  \n"
    f"ROC-AUC test : **{m['roc_auc']:.3f}**  \n"
    f"F2 test : **{m['f2']:.3f}**  \n"
    f"Rappel au seuil F2 : **{m['recall']:.1%}**  \n"
    f"Entraîné le {meta['date_entrainement'][:10]}"
)
st.sidebar.divider()
st.sidebar.subheader("Seuils de décision")
st.sidebar.caption("Choisis par validation croisée dans le notebook 01.")
st.sidebar.markdown(
    f"- **Seuil F2 = {SEUIL_F2:.2f}** : en dessous, aucune action  \n"
    f"- **Seuil F0,5 = {SEUIL_F05:.2f}** : au-dessus, action contraignante"
)
for zone, (action, description) in pr.ACTIONS.items():
    st.sidebar.markdown(f"**{zone}** : {action}")

# ---------------------------------------------------------------------------
# En-tête et onglets
# ---------------------------------------------------------------------------
st.title("Prévoir les annulations, décider quoi faire")
st.markdown(
    "Le modèle **prédit** la probabilité d'annulation de chaque réservation. L'application en tire des "
    "**décisions** chiffrées à partir des données : quelle action mener, ce que coûtent les annulations, "
    "ce qu'un acompte permettrait de récupérer, combien de chambres surréserver."
)
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔎 Réservation", "📋 Lot de réservations", "💶 Pertes et acomptes", "🛏️ Surréservation"]
)

# ===========================================================================
# 1. Réservation unique
# ===========================================================================
with tab1:
    st.subheader("Évaluer une réservation")
    pays = meta["pays"]
    with st.form("reservation"):
        c1, c2, c3, c4 = st.columns(4)
        hotel = c1.selectbox("Hôtel", ["City Hotel", "Resort Hotel"])
        lead_time = c1.number_input("Délai avant l'arrivée (jours)", 0, 800, 120)
        mois = c1.selectbox("Mois d'arrivée", config.MONTHS, index=7)
        jour = c1.number_input("Jour d'arrivée", 1, 31, 15)

        weekend = c2.number_input("Nuits de fin de semaine", 0, 20, 2)
        semaine = c2.number_input("Nuits de semaine", 0, 50, 3)
        adultes = c2.number_input("Adultes", 0, 10, 2)
        enfants = c2.number_input("Enfants", 0, 10, 0)
        bebes = c2.number_input("Bébés", 0, 10, 0)

        adr = c3.number_input("Tarif journalier moyen (€)", 0.0, 1000.0, 110.0, step=5.0)
        depot = c3.selectbox("Type de dépôt", ["No Deposit", "Non Refund", "Refundable"])
        segment = c3.selectbox("Segment de marché", ["Online TA", "Offline TA/TO", "Direct", "Groups",
                                                      "Corporate", "Complementary", "Aviation"])
        canal = c3.selectbox("Canal de distribution", ["TA/TO", "Direct", "Corporate", "GDS"])
        type_client = c3.selectbox("Type de client", ["Transient", "Transient-Party", "Contract", "Group"])

        pays_choisi = c4.selectbox("Pays", pays, index=pays.index("PRT") if "PRT" in pays else 0)
        repas = c4.selectbox("Repas", ["BB", "HB", "FB", "SC"])
        chambre = c4.selectbox("Type de chambre réservée", list("ABCDEFGHL"))
        demandes = c4.number_input("Demandes spéciales", 0, 5, 0)
        parking = c4.number_input("Places de stationnement", 0, 3, 0)

        with st.expander("Historique du client et autres informations"):
            e1, e2, e3 = st.columns(3)
            fidele = e1.checkbox("Client déjà venu")
            annul_passees = e1.number_input("Annulations passées", 0, 30, 0)
            sejours_passes = e2.number_input("Séjours passés non annulés", 0, 80, 0)
            modifs = e2.number_input("Modifications de la réservation", 0, 20, 0)
            attente = e3.number_input("Jours en liste d'attente", 0, 400, 0)
            via_agence = e3.checkbox("Réservé via une agence", value=True)
            via_entreprise = e3.checkbox("Payé par une entreprise")
        st.form_submit_button("Évaluer", type="primary")

    # Numéro de semaine déduit de la date (année de référence 2017)
    semaine_annee = int(pd.Timestamp(2017, config.MONTHS.index(mois) + 1, min(int(jour), 28)).isocalendar().week)
    reservation = pd.DataFrame([{
        "hotel": hotel, "lead_time": lead_time, "arrival_date_month": mois,
        "arrival_date_week_number": semaine_annee, "arrival_date_day_of_month": jour,
        "stays_in_weekend_nights": weekend, "stays_in_week_nights": semaine, "adults": adultes,
        "children": enfants, "babies": bebes, "meal": repas, "country": pays_choisi,
        "market_segment": segment, "distribution_channel": canal, "is_repeated_guest": int(fidele),
        "previous_cancellations": annul_passees, "previous_bookings_not_canceled": sejours_passes,
        "reserved_room_type": chambre, "booking_changes": modifs, "deposit_type": depot,
        "agent": 9.0 if via_agence else np.nan, "company": 40.0 if via_entreprise else np.nan,
        "days_in_waiting_list": attente, "customer_type": type_client, "adr": adr,
        "required_car_parking_spaces": parking, "total_of_special_requests": demandes,
    }])

    p = float(modele.predict_proba(reservation)[0, 1])
    revenu = float(pr.revenue_at_risk([adr], [weekend + semaine])[0])
    zone = pr.risk_level(p, SEUIL_F2, SEUIL_F05)
    action, description = pr.action_for(zone, depot)
    perte = float(pr.expected_loss([p], [revenu], [depot])[0])

    g1, g2 = st.columns([1, 2])
    with g1:
        jauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": " %", "valueformat": ".1f"},
            title={"text": f"Risque {zone.lower()}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COULEURS_RISQUE[zone]},
                "steps": [
                    {"range": [0, SEUIL_F2 * 100], "color": "#e8f4f2"},
                    {"range": [SEUIL_F2 * 100, SEUIL_F05 * 100], "color": "#fbf3dc"},
                    {"range": [SEUIL_F05 * 100, 100], "color": "#f8e1de"},
                ],
            },
        ))
        jauge.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(jauge, width="stretch")
    with g2:
        k1, k2, k3 = st.columns(3)
        k1.metric("Probabilité d'annulation", f"{p:.1%}")
        k2.metric("Revenu du séjour", euro(revenu), help="Tarif journalier x nuits")
        k3.metric("Perte espérée", euro(perte),
                  help="Probabilité x revenu ; 0 € si la réservation est non remboursable (séjour déjà payé)")
        st.success(f"**Action recommandée : {action}**  \n{description}")

    st.markdown("##### Sensibilité au délai de réservation")
    st.caption("Réponse du modèle si seul le délai change (association, pas effet causal).")
    grille = np.arange(0, 401, 10)
    variantes = pd.concat([reservation] * len(grille), ignore_index=True).assign(lead_time=grille)
    fig = px.line(x=grille, y=modele.predict_proba(variantes)[:, 1],
                  labels={"x": "Délai avant l'arrivée (jours)", "y": "Probabilité d'annulation"})
    fig.add_vline(x=lead_time, line_dash="dash", line_color=ROUGE)
    fig.add_hline(y=SEUIL_F2, line_dash="dot", line_color=GRIS, annotation_text="seuil F2")
    fig.add_hline(y=SEUIL_F05, line_dash="dot", line_color=ROUGE, annotation_text="seuil F0,5")
    fig.update_traces(line_color=BLEU)
    fig.update_layout(height=300, yaxis_tickformat=".0%", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# ===========================================================================
# 2. Lot de réservations
# ===========================================================================
with tab2:
    st.subheader("Prioriser un ensemble de réservations")
    source = st.radio("Source", ["Échantillon de l'ensemble de test", "Importer un fichier CSV"], horizontal=True)
    lot = None
    if source == "Importer un fichier CSV":
        st.caption("Le fichier doit contenir les colonnes brutes du jeu hotels.csv.")
        fichier = st.file_uploader("Fichier CSV", type="csv")
        if fichier is None:
            st.info("Importez un fichier pour lancer l'analyse.")
        else:
            importe = pd.read_csv(fichier, na_values=["NULL", ""])
            manquantes = sorted(set(config.RAW_INPUT_FEATURES) - set(importe.columns))
            if manquantes:
                st.error(f"Colonnes manquantes : {manquantes}")
            else:
                lot = importe.reset_index(drop=True)
    else:
        n = st.slider("Nombre de réservations", 50, 2000, 500, 50)
        _, test = charger_test()
        lot = test.sample(n, random_state=config.RANDOM_STATE).reset_index(drop=True)

    # Pas de st.stop() : il interromprait aussi l'affichage des onglets suivants
    if lot is not None:
        proba_lot = modele.predict_proba(lot[config.RAW_INPUT_FEATURES])[:, 1]
        nuits_lot = lot["stays_in_weekend_nights"] + lot["stays_in_week_nights"]
        reco = pr.recommend_actions(proba_lot, pr.revenue_at_risk(lot["adr"], nuits_lot), SEUIL_F2, SEUIL_F05,
                                    deposit_type=lot["deposit_type"])
        sortie = pd.concat(
            [lot[["hotel", "arrival_date_month", "arrival_date_day_of_month", "lead_time", "adr",
                  "market_segment", "deposit_type", "country"]], reco], axis=1
        ).sort_values("perte_esperee", ascending=False)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Réservations", len(sortie))
        k2.metric("Annulations attendues", f"{proba_lot.sum():.0f}", f"{proba_lot.mean():.1%} du lot", delta_color="off")
        k3.metric("Perte espérée totale", euro(reco["perte_esperee"].sum()))
        k4.metric("Zone de risque élevé", int((reco["niveau_risque"] == "Élevé").sum()))

        c1, c2 = st.columns(2)
        hist = px.histogram(sortie, x="proba_annulation", color="niveau_risque", nbins=40,
                            color_discrete_map=COULEURS_RISQUE, labels={"proba_annulation": "Probabilité d'annulation"})
        hist.update_layout(height=320, bargap=0.05, legend_title_text="Risque", margin=dict(t=30, b=10))
        c1.plotly_chart(hist, width="stretch")
        agg = sortie.groupby("action_recommandee").size().rename("reservations").reset_index()
        barres = px.bar(agg, x="reservations", y="action_recommandee", orientation="h", text="reservations",
                        labels={"reservations": "Réservations", "action_recommandee": ""},
                        color_discrete_sequence=[BLEU])
        barres.update_layout(height=320, margin=dict(t=30, b=10))
        c2.plotly_chart(barres, width="stretch")

        st.markdown("##### Réservations à traiter en priorité (perte espérée décroissante)")
        st.dataframe(
            sortie, hide_index=True, width="stretch",
            column_config={
                "proba_annulation": st.column_config.ProgressColumn("Probabilité", format="percent", min_value=0, max_value=1),
                "revenu_en_jeu": st.column_config.NumberColumn("Revenu du séjour (€)", format="%.0f"),
                "perte_esperee": st.column_config.NumberColumn("Perte espérée (€)", format="%.0f"),
            },
        )
        st.download_button("Télécharger les recommandations (CSV)", sortie.to_csv(index=False).encode("utf-8"),
                           "recommandations_annulations.csv", "text/csv")

# ===========================================================================
# 3. Pertes et acomptes
# ===========================================================================
with tab3:
    st.subheader("Combien coûtent les annulations, et que rapporterait un acompte ?")
    with st.spinner("Calcul de l'occupation nuit par nuit..."):
        _, test = charger_test()
        revente, capacite = charger_revente()
    pertes = pr.loss_table(pr.cancellation_timing(charger_toutes()), revente)

    k1, k2, k3 = st.columns(3)
    k1.metric("Manque à gagner brut", euro(pertes["revenu_brut"].sum()), help="Séjours annulés sans dépôt")
    k2.metric("Manque à gagner net", euro(pertes["perte_nette"].sum()), help="Après revente des chambres")
    k3.metric("Capacité estimée", f"{int(capacite.get('City Hotel', 0))} / {int(capacite.get('Resort Hotel', 0))}",
              help="Chambres City Hotel / Resort Hotel (99e centile de l'occupation observée)")

    tableau = pertes[["annulations", "part_annulations", "part_groupes", "revenu_brut", "taux_revente", "perte_nette"]]
    tableau = tableau.rename_axis("Délai avant l'arrivée")  # libellé lisible au lieu de delay_category
    st.dataframe(
        tableau, width="stretch",
        column_config={
            "part_annulations": st.column_config.NumberColumn("Part des annulations", format="percent"),
            "part_groupes": st.column_config.NumberColumn("Part de groupes", format="percent"),
            "revenu_brut": st.column_config.NumberColumn("Revenu brut (€)", format="%.0f"),
            "taux_revente": st.column_config.NumberColumn("Taux de revente", format="percent"),
            "perte_nette": st.column_config.NumberColumn("Perte nette (€)", format="%.0f"),
        },
    )
    st.caption("Délai = jours entre l'annulation et la date d'arrivée prévue. Taux de revente estimé à partir "
               "de l'occupation réelle de chaque nuit (méthode détaillée dans le notebook 02).")

    st.markdown("#### Simulateur d'acompte")
    st.caption("Appliqué aux réservations SANS dépôt de l'ensemble de test signalées par le modèle (une réservation "
               "non remboursable est déjà payée). La perte nette de départ est la même pour tous les seuils : celle de "
               "toutes les annulations sans dépôt. Le niveau d'acompte est un choix de l'hôtel ; tout le reste est "
               "calculé à partir des données.")
    s1, s2, s3, s4 = st.columns(4)
    type_acompte = s1.radio("Type d'acompte", ["Part du séjour", "Nombre de nuits"])
    if type_acompte == "Part du séjour":
        valeur = s2.slider("Part du séjour", 0.05, 1.0, 0.20, 0.05, format="%.2f")
        kind = "part"
    else:
        valeur = s2.slider("Nombre de nuits", 1, 5, 1)
        kind = "nuits"
    remboursable = s3.slider("Remboursable jusqu'à (jours avant l'arrivée)", 0, 90, 0, 5,
                             help="0 = jamais remboursé")
    # Seuils exacts du notebook 01 (non arrondis) pour retrouver les mêmes chiffres,
    # ou seuil libre pour explorer
    choix_seuil = s4.radio(
        "Seuil de probabilité",
        [f"F0,5 = {SEUIL_F05:.3f} (précision)", f"F2 = {SEUIL_F2:.3f} (rappel)", "Autre seuil"],
    )
    if choix_seuil.startswith("F0,5"):
        seuil = SEUIL_F05
    elif choix_seuil.startswith("F2"):
        seuil = SEUIL_F2
    else:
        seuil = s4.slider("Seuil choisi", 0.05, 0.95, 0.50, 0.01)

    politique = {"choisie": (kind, valeur, remboursable or None)}
    res = pr.deposit_scenarios(test, test["proba"] >= seuil, revente, politique).iloc[0]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Acompte conservé", euro(res["acompte_conserve"]),
              f"{res['part_perte_couverte']:.0%} de la perte nette", delta_color="off")
    k2.metric("Perte nette restante", euro(res["perte_nette_restante"]),
              f"sur {euro(res['perte_nette_totale'])} au départ", delta_color="off")
    k3.metric("Annulations visées", int(res["annulations_signalees"]),
              f"{res['part_annulations_visees']:.0%} de toutes les annulations", delta_color="off")
    k4.metric("Clients fidèles touchés", int(res["fideles_exposes"]),
              f"{res['part_fideles_touches']:.1%} de tous les fidèles", delta_color="off")
    k5.metric("Fuite max. acceptable", f"{res['taux_fuite_max_acceptable']:.1%}",
              help="Part des fidèles touchés qui peuvent renoncer avant que l'acompte coûte plus qu'il ne rapporte")

    # Comparaison des seuils sur une échelle commune : le gain net selon la part des fidèles
    # qui renoncent. Ce taux décrit la réaction des clients, pas le seuil : même axe pour tous.
    st.markdown("##### Comparer les seuils : gain net selon la part des fidèles qui renoncent")
    st.caption("Gain net = acompte conservé - revenu des clients fidèles perdus. La part des fidèles qui renoncent "
               "devant un acompte est inconnue (seul un test A/B la mesurerait) ; elle est la même quel que soit le "
               "seuil, ce qui permet de comparer les seuils sur le même axe.")
    seuils_compares = {f"F0,5 = {SEUIL_F05:.3f}": SEUIL_F05, f"F2 = {SEUIL_F2:.3f}": SEUIL_F2}
    if not (choix_seuil.startswith("F0,5") or choix_seuil.startswith("F2")):
        seuils_compares[f"Seuil choisi = {seuil:.2f}"] = seuil
    scenarios = {nom: pr.deposit_scenarios(test, test["proba"] >= s, revente, politique).iloc[0]
                 for nom, s in seuils_compares.items()}
    taux_fuite = np.linspace(0, 0.6, 61)
    fig = go.Figure()
    for (nom, sc), couleur in zip(scenarios.items(), [ROUGE, BLEU, VERT]):
        fig.add_trace(go.Scatter(x=taux_fuite, y=pr.net_gain_by_leak_rate(sc, taux_fuite), name=nom,
                                 line_color=couleur))
    fig.add_hline(y=0, line_color=GRIS)
    sc_f05, sc_f2 = list(scenarios.values())[:2]
    bascule = pr.switch_leak_rate(sc_f2, sc_f05)
    if 0 < bascule < 0.6:
        fig.add_vline(x=bascule, line_dash="dash", line_color=GRIS,
                      annotation_text=f"bascule : {bascule:.1%}", annotation_position="bottom right")
    fig.update_layout(height=340, xaxis_title="Part des clients fidèles touchés qui renoncent",
                      yaxis_title="Gain net (€)", xaxis_tickformat=".0%", margin=dict(t=30),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    if 0 < bascule < 1:
        st.markdown(
            f"Avec cette politique, **F2 rapporte plus tant que moins de {bascule:.1%} des fidèles renoncent** ; "
            f"au-delà, **F0,5 rapporte plus**. L'acompte devient perdant à partir de "
            f"{sc_f2['taux_fuite_max_acceptable']:.0%} de fuite au seuil F2 et de "
            f"{sc_f05['taux_fuite_max_acceptable']:.0%} au seuil F0,5."
        )

# ===========================================================================
# 4. Surréservation
# ===========================================================================
with tab4:
    st.subheader("Combien de réservations accepter au-delà de la capacité ?")
    st.markdown(
        "Le nombre de clients présents suit une **loi de Poisson binomiale** (chaque réservation a sa propre "
        "probabilité d'annulation). On retient le plus grand nombre de réservations supplémentaires tel que le "
        "**risque de devoir reloger au moins un client** reste sous le niveau choisi."
    )
    _, test = charger_test()
    c1, c2 = st.columns(2)
    hotel_choisi = c1.selectbox("Hôtel ", ["City Hotel", "Resort Hotel"], key="sur_hotel")
    sous = test[test["hotel"] == hotel_choisi]
    dates = (sous.groupby(["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).size()
             .sort_values(ascending=False).head(30))
    # Chaque clé de l'index est un triplet (année, mois, jour)
    libelles = [f"{cast(tuple, cle)[2]} {cast(tuple, cle)[1]} {cast(tuple, cle)[0]} ({n} réservations)"
                for cle, n in dates.items()]
    choix = c2.selectbox("Date d'arrivée (30 dates les plus chargées du test)", range(len(libelles)),
                         format_func=lambda i: libelles[i])
    a_, mo_, d_ = cast(tuple, dates.index[choix])
    carnet = sous[(sous["arrival_date_year"] == a_) & (sous["arrival_date_month"] == mo_)
                  & (sous["arrival_date_day_of_month"] == d_)]

    c1, c2, c3 = st.columns(3)
    cap = c1.number_input("Capacité (chambres)", 1, 500, len(carnet), help="Par défaut : hôtel complet")
    risque_max = c2.slider("Risque de relogement accepté", 0.01, 0.30, 0.05, 0.01, format="%.2f")
    p_nouvelles = c3.slider("Probabilité d'annulation des nouvelles réservations", 0.0, 0.9,
                            float(round(carnet["proba"].mean(), 2)), 0.01)

    plan = pr.overbooking_plan(carnet["proba"], int(cap), p_nouvelles, max_extra=max(int(cap), 10))
    retenu = pr.max_safe_overbooking(plan, risque_max)
    base = plan.iloc[0]
    k = int(retenu["reservations_supplementaires"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Réservations supplémentaires", f"+{k}")
    k2.metric("Taux d'occupation espéré", f"{retenu['taux_occupation_espere']:.1%}",
              f"{(retenu['taux_occupation_espere'] - base['taux_occupation_espere']) * 100:+.1f} pts")
    k3.metric("Risque de relogement", f"{retenu['prob_au_moins_un_reloge']:.1%}")
    k4.metric("Annulations prévues / observées", f"{carnet['proba'].sum():.1f} / {int(carnet['is_canceled'].sum())}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plan["reservations_supplementaires"], y=plan["taux_occupation_espere"],
                             name="Taux d'occupation espéré", line_color=BLEU))
    fig.add_trace(go.Scatter(x=plan["reservations_supplementaires"], y=plan["prob_au_moins_un_reloge"],
                             name="Risque de relogement", line_color=ROUGE))
    fig.add_hline(y=risque_max, line_dash="dot", line_color=ROUGE)
    fig.add_vline(x=k, line_dash="dash", line_color=GRIS, annotation_text=f"+{k}")
    fig.update_layout(height=350, xaxis_title="Réservations acceptées au-delà de la capacité",
                      yaxis_tickformat=".0%", margin=dict(t=30), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")
    st.caption("Les dates proviennent de l'ensemble de test (20 % des réservations) : le carnet est donc réduit, "
               "mais le principe est identique sur un carnet complet.")
