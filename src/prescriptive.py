"""Analyse prescriptive : passer de « qui va annuler ? » à « que faut-il faire ? ».

Le modèle prédictif donne une probabilité d'annulation p. Ce module la
transforme en décisions chiffrées, à partir d'hypothèses de coûts que
l'utilisateur peut modifier (dans l'application Streamlit notamment).

Trois outils de décision
------------------------
1. Seuil de décision optimal : au lieu du seuil arbitraire de 0,5, on choisit
   le seuil qui minimise le coût total des erreurs :
       coût = FN x coût d'une annulation non anticipée
            + FP x coût d'une intervention inutile
2. Action recommandée par réservation : parmi un catalogue d'actions
   (rappel, offre, dépôt...), on retient celle qui maximise le gain net
   espéré :
       gain = p x revenu x efficacité            (annulations évitées)
            - coût de l'action
            - (1 - p) x revenu x désistement     (bons clients perdus, car
                                                  une demande de prépaiement
                                                  peut faire fuir un client
                                                  qui serait venu)
3. Surréservation optimale pour une date : à partir des probabilités de
   chaque réservation, on calcule la distribution exacte du nombre de clients
   présents (loi de Poisson binomiale) et le nombre de réservations
   supplémentaires qui maximise le profit espéré :
       profit = revenu x E[chambres occupées] - pénalité x E[clients relogés]

Limite importante : le modèle mesure une association, pas un effet causal.
Les efficacités des actions sont des HYPOTHÈSES métier à valider par un
test A/B ; elles ne sont pas estimées à partir des données.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Seuil de décision optimal
# ---------------------------------------------------------------------------


def cost_by_threshold(y_true, y_proba, cost_fn: float, cost_fp: float, thresholds=None) -> pd.DataFrame:
    """Coût total des erreurs pour chaque seuil candidat."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = np.linspace(0.05, 0.95, 91) if thresholds is None else np.asarray(thresholds)
    rows = []
    for t in thresholds:
        pred = y_proba >= t
        fn = int(((~pred) & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        tp = int((pred & (y_true == 1)).sum())
        rows.append(
            {
                "seuil": round(float(t), 3),
                "faux_negatifs": fn,
                "faux_positifs": fp,
                "vrais_positifs": tp,
                "rappel": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "cout_total": fn * cost_fn + fp * cost_fp,
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(y_true, y_proba, cost_fn: float, cost_fp: float) -> tuple[float, pd.DataFrame]:
    table = cost_by_threshold(y_true, y_proba, cost_fn, cost_fp)
    best = table.loc[table["cout_total"].idxmin()]
    return float(best["seuil"]), table


# ---------------------------------------------------------------------------
# 2. Action recommandée par réservation
# ---------------------------------------------------------------------------


@dataclass
class Action:
    nom: str
    cout: float         # coût unitaire de l'action (€ par réservation)
    efficacite: float   # part supposée des annulations évitées (0 à 1)
    desistement: float  # part supposée des clients fidèles perdus à cause de l'action (0 à 1)
    description: str


def default_actions() -> list[Action]:
    """Catalogue d'actions par défaut. Valeurs = hypothèses modifiables."""
    return [
        Action("Aucune action", 0.0, 0.00, 0.00, "Confirmation automatique standard."),
        Action(
            "Rappel personnalisé",
            2.0,
            0.10,
            0.00,
            "Courriel et SMS de rappel avant l'arrivée, avec lien de confirmation.",
        ),
        Action(
            "Offre de fidélisation",
            15.0,
            0.25,
            0.00,
            "Avantage conditionné à la confirmation (petit-déjeuner, surclassement, départ tardif).",
        ),
        Action(
            "Demande de prépaiement partiel",
            5.0,
            0.35,
            0.05,
            "Acompte ou passage à un tarif non remboursable avec remise.",
        ),
    ]


def revenue_at_risk(adr, total_nights, resale_rate: float = 0.3):
    """Revenu perdu si la réservation est annulée.

    resale_rate : part des chambres annulées que l'hôtel parvient à revendre
    (une annulation tardive est rarement revendue).
    """
    return np.asarray(adr) * np.maximum(np.asarray(total_nights), 1) * (1 - resale_rate)


def recommend_actions(proba, revenue, actions: list[Action] | None = None) -> pd.DataFrame:
    """Pour chaque réservation, choisit l'action au gain net espéré maximal."""
    actions = actions or default_actions()
    proba = np.asarray(proba, dtype=float)
    revenue = np.asarray(revenue, dtype=float)
    gains = np.column_stack(
        [proba * revenue * a.efficacite - a.cout - (1 - proba) * revenue * a.desistement for a in actions]
    )
    best = gains.argmax(axis=1)
    return pd.DataFrame(
        {
            "proba_annulation": proba,
            "revenu_en_jeu": revenue,
            "perte_esperee": proba * revenue,
            "action_recommandee": [actions[i].nom for i in best],
            "gain_net_espere": gains[np.arange(len(best)), best],
        }
    )


def risk_level(p: float, low: float = 0.3, high: float = 0.6) -> str:
    if p >= high:
        return "Élevé"
    if p >= low:
        return "Moyen"
    return "Faible"


# ---------------------------------------------------------------------------
# 3. Surréservation optimale
# ---------------------------------------------------------------------------


def show_distribution(p_cancel) -> np.ndarray:
    """Loi exacte du nombre de clients présents (Poisson binomiale).

    Chaque réservation i se présente avec la probabilité (1 - p_i), de façon
    indépendante. Calcul par programmation dynamique (convolution successive).
    dist[k] = P(exactement k clients présents).
    """
    dist = np.array([1.0])
    for p in np.asarray(p_cancel, dtype=float):
        q = 1.0 - p  # probabilité de présence
        new = np.zeros(len(dist) + 1)
        new[:-1] += dist * p
        new[1:] += dist * q
        dist = new
    return dist


def overbooking_plan(
    p_cancel_existing,
    capacity: int,
    p_cancel_new: float,
    room_revenue: float,
    walk_cost: float,
    max_extra: int = 40,
) -> pd.DataFrame:
    """Profit espéré selon le nombre k de réservations acceptées au-delà de la capacité.

    p_cancel_existing : probabilités prédites des réservations déjà au carnet
    capacity          : nombre de chambres disponibles
    p_cancel_new      : probabilité d'annulation supposée des réservations ajoutées
    room_revenue      : revenu d'une chambre occupée (€ pour le séjour)
    walk_cost         : pénalité si un client présent n'a pas de chambre
                        (relogement, dédommagement, atteinte à la réputation)
    """
    p_cancel_existing = np.asarray(p_cancel_existing, dtype=float)
    base_overbook = max(len(p_cancel_existing) - capacity, 0)
    rows = []
    for k in range(0, max_extra + 1):
        probs = np.concatenate([p_cancel_existing, np.full(k, p_cancel_new)])
        dist = show_distribution(probs)
        shows = np.arange(len(dist))
        occupied = np.minimum(shows, capacity)
        walked = np.maximum(shows - capacity, 0)
        exp_occ = float((dist * occupied).sum())
        exp_walk = float((dist * walked).sum())
        rows.append(
            {
                "reservations_supplementaires": k,
                "reservations_totales": len(probs),
                "surreservation_totale": base_overbook + k,
                "presents_esperes": float((dist * shows).sum()),
                "chambres_occupees_esperees": exp_occ,
                "taux_occupation_espere": exp_occ / capacity if capacity else 0.0,
                "clients_reloges_esperes": exp_walk,
                "prob_au_moins_un_reloge": float(dist[shows > capacity].sum()),
                "profit_espere": room_revenue * exp_occ - walk_cost * exp_walk,
            }
        )
    return pd.DataFrame(rows)


def best_overbooking(plan: pd.DataFrame) -> pd.Series:
    return plan.loc[plan["profit_espere"].idxmax()]
