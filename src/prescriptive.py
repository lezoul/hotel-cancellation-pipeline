"""Analyse prescriptive : passer de « qui va annuler ? » à « que faut-il faire ? ».

Utilisé par notebooks/02_analyse_prescriptive.ipynb et app/app.py.

Principe : aucun chiffre inventé
--------------------------------
Toutes les grandeurs sont calculées à partir des données :
    * le moment des annulations (date d'annulation - date d'arrivée) ;
    * le taux de revente des chambres annulées, estimé nuit par nuit à partir
      de l'occupation réelle des hôtels ;
    * le manque à gagner brut et net ;
    * les seuils de décision (F2, F0,5) choisis par validation croisée.
Seuls les NIVEAUX D'ACOMPTE sont des choix de l'hôtel : ils sont présentés
comme des scénarios à comparer, jamais comme des faits.

Ce qu'on ne peut PAS calculer : combien de clients fidèles renonceraient à
réserver si on leur demandait un acompte (ils n'apparaissent pas dans les
données). On calcule à la place le taux de fuite MAXIMAL acceptable : au-delà,
l'acompte coûte plus qu'il ne rapporte.

Remarque sur la fuite de données : l'analyse du moment des annulations
utilise reservation_status_date, colonne exclue du modèle prédictif. Ici
c'est légitime : on fait un bilan après coup, pas une prédiction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.data_loader import load_raw_data

# ===========================================================================
# 0. Tranches de délai d'annulation (5 catégories)
# ===========================================================================
# Délai = nombre de jours entre l'annulation et la date d'arrivée prévue.
# Bornes en jours : [0, 7[, [7, 30[, [30, 60[, [60, 90[, [90, +inf[
DELAY_BINS = [-np.inf, 7, 30, 60, 90, np.inf]
DELAY_LABELS = [
    "moins d'une semaine",
    "1 semaine à 1 mois",
    "1 à 2 mois",
    "2 à 3 mois",
    "plus de 3 mois",
]

MONTH_NUMBER = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}


def delay_category(delay_days) -> pd.Series:
    """Range chaque délai (en jours) dans l'une des 5 tranches.

    right=False : la borne de gauche est incluse (7 jours -> « 1 semaine à 1 mois »).
    """
    return pd.cut(pd.Series(delay_days), bins=DELAY_BINS, labels=DELAY_LABELS, right=False)


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute date d'arrivée, date de statut, délai d'annulation, nuits et revenu du séjour."""
    df = df.copy()
    # pd.to_datetime sur un DataFrame year / month / day : une date par ligne
    df["arrival_date"] = pd.to_datetime(
        pd.DataFrame({
            "year": df["arrival_date_year"],
            "month": df["arrival_date_month"].map(MONTH_NUMBER),
            "day": df["arrival_date_day_of_month"],
        })
    )
    df["status_date"] = pd.to_datetime(df["reservation_status_date"])
    # Délai entre l'annulation et l'arrivée ; une absence (No-Show) donne 0 ou moins
    df["cancel_delay"] = (df["arrival_date"] - df["status_date"]).dt.days.clip(lower=0)
    df["nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["stay_revenue"] = revenue_at_risk(df["adr"], df["nights"])
    df["delay_category"] = delay_category(df["cancel_delay"]).values
    return df


def revenue_at_risk(adr, total_nights):
    """Revenu du séjour : tarif journalier x nombre de nuits (0 nuit compte pour 1)."""
    return np.asarray(adr, dtype=float) * np.maximum(np.asarray(total_nights, dtype=float), 1)


def load_all_bookings(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Toutes les réservations, DOUBLONS COMPRIS, avec dates (bilan financier).

    Pour compter des chambres et de l'argent, chaque ligne compte : deux lignes
    identiques sont souvent deux vraies chambres réservées par un même groupe.
    Seules les lignes impossibles (aucun client, tarif négatif ou > 5 000 €)
    sont retirées, comme dans le nettoyage de base.
    (Le modèle, lui, utilise les données sans doublons, pour que l'évaluation
    ne soit pas faussée par des lignes identiques dans l'entraînement et le test.)
    """
    if raw is None:
        raw = load_raw_data()
    guests = raw["adults"] + raw["children"].fillna(0) + raw["babies"]
    raw = raw[(guests > 0) & (raw["adr"] >= 0) & (raw["adr"] <= 5000)].reset_index(drop=True)
    return add_dates(raw)


# ===========================================================================
# 1. Quand les clients annulent-ils ?
# ===========================================================================
def cancellation_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Répartition des annulations sans dépôt par tranche de délai.

    df : réservations avec add_dates() déjà appliqué.
    Colonnes : nombre d'annulations, part du total, part de réservations de
    groupes, revenu brut des séjours annulés.
    """
    c = df[(df["is_canceled"] == 1) & (df["deposit_type"] == "No Deposit")]
    table = c.groupby("delay_category", observed=False).agg(
        annulations=("hotel", "size"),
        part_groupes=("market_segment", lambda s: (s == "Groups").mean()),
        nuitees=("nights", "sum"),
        revenu_brut=("stay_revenue", "sum"),
    )
    table.insert(1, "part_annulations", table["annulations"] / table["annulations"].sum())
    return table


# ===========================================================================
# 2. Taux de revente estimé à partir de l'occupation réelle
# ===========================================================================
def _explode_nights(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par nuit occupée (ou réservée) : (index de la réservation, hôtel, nuit)."""
    nights = df["nights"].clip(lower=0).to_numpy()
    idx = np.repeat(df.index.to_numpy(), nights)                       # index répété autant de fois que de nuits
    offsets = np.concatenate([np.arange(n) for n in nights]) if len(df) else np.array([], dtype=int)
    night = np.repeat(df["arrival_date"].to_numpy(), nights) + pd.to_timedelta(offsets, unit="D")
    return pd.DataFrame({"i": idx, "hotel": np.repeat(df["hotel"].to_numpy(), nights), "night": night})


def estimate_resale(raw: pd.DataFrame | None = None, capacity_quantile: float = 0.99):
    """Estime, nuit par nuit, la part des chambres annulées qui ont été revendues.

    Méthode
    -------
    1. Occupation réelle de chaque nuit = chambres des séjours réalisés.
    2. Capacité de chaque hôtel = 99e centile de l'occupation observée
       (le nombre de chambres n'est pas dans les données).
    3. Une nuit donnée, les chambres annulées sont perdues dans la limite
       des chambres restées vides ; le reste est considéré comme revendu :
           perdues = min(chambres annulées, capacité - occupation)
           taux de revente de la nuit = 1 - perdues / chambres annulées
    4. Chaque annulation reçoit la moyenne des taux de ses nuits.

    Données AVEC doublons : pour l'occupation, deux lignes identiques peuvent
    être deux vraies chambres réservées par un même groupe.

    Renvoie (taux par annulation, capacités estimées, table des nuits).
    """
    raw = load_all_bookings(raw)  # doublons conservés, lignes impossibles retirées

    # 1. Occupation réelle par hôtel et par nuit
    realised = _explode_nights(raw[raw["is_canceled"] == 0])
    occupancy = realised.groupby(["hotel", "night"]).size().rename("occupees")

    # 2. Capacité estimée
    capacity = occupancy.groupby(level="hotel").quantile(capacity_quantile).round()

    # 3. Chambres annulées (sans dépôt) par nuit
    cancelled = raw[(raw["is_canceled"] == 1) & (raw["deposit_type"] == "No Deposit")]
    cancelled_nights = _explode_nights(cancelled)
    cancelled_count = cancelled_nights.groupby(["hotel", "night"]).size().rename("annulees")

    nights = pd.concat([occupancy, cancelled_count], axis=1).fillna(0)
    nights["capacite"] = nights.index.get_level_values("hotel").map(capacity.to_dict())
    nights["vides"] = (nights["capacite"] - nights["occupees"]).clip(lower=0)
    nights["perdues"] = np.minimum(nights["annulees"], nights["vides"])
    nights["taux_revente"] = np.where(
        nights["annulees"] > 0, 1 - nights["perdues"] / nights["annulees"].where(nights["annulees"] > 0, 1), np.nan
    )

    # 4. Taux moyen par annulation (moyenne sur ses nuits)
    cancelled_nights = cancelled_nights.join(nights["taux_revente"], on=["hotel", "night"])
    per_booking = cancelled_nights.groupby("i")["taux_revente"].mean()
    cancelled = cancelled.assign(taux_revente=per_booking)
    return cancelled, capacity, nights


def resale_by_delay(cancelled: pd.DataFrame) -> pd.Series:
    """Taux de revente moyen par tranche de délai (pondéré par le revenu du séjour)."""
    c = cancelled.dropna(subset=["taux_revente"])
    weighted = (c["taux_revente"] * c["stay_revenue"]).groupby(c["delay_category"], observed=False).sum()
    return (weighted / c.groupby("delay_category", observed=False)["stay_revenue"].sum()).rename("taux_revente")


def loss_table(timing: pd.DataFrame, resale: pd.Series) -> pd.DataFrame:
    """Manque à gagner brut et net (après revente) par tranche de délai."""
    table = timing.join(resale)
    table["perte_nette"] = table["revenu_brut"] * (1 - table["taux_revente"])
    table["part_perte_nette"] = table["perte_nette"] / table["perte_nette"].sum()
    return table


# ===========================================================================
# 3. Scénarios d'acompte
# ===========================================================================
DEPOSIT_POLICIES = {
    # nom : (type, valeur, remboursable jusqu'à N jours avant l'arrivée ou None)
    "Première nuit": ("nuits", 1, None),
    "20 % du séjour": ("part", 0.20, None),
    "50 % du séjour": ("part", 0.50, None),
    "50 %, remboursable jusqu'à 30 jours": ("part", 0.50, 30),
    "100 %, remboursable jusqu'à 30 jours": ("part", 1.00, 30),
}


def deposit_amount(adr, nights, kind: str, value: float):
    """Montant de l'acompte selon la politique : un nombre de nuits ou une part du séjour."""
    revenue = revenue_at_risk(adr, nights)
    if kind == "nuits":
        return np.minimum(np.asarray(adr, dtype=float) * value, revenue)  # pas plus que le séjour
    return revenue * value


def deposit_scenarios(bookings: pd.DataFrame, flagged, resale: pd.Series, policies: dict | None = None) -> pd.DataFrame:
    """Bilan de chaque politique d'acompte appliquée aux réservations signalées.

    bookings : TOUTES les réservations étudiées (avec add_dates) contenant is_canceled
    flagged  : masque booléen des réservations signalées par le modèle (même longueur)
    resale   : taux de revente par tranche de délai (resale_by_delay)

    Deux règles rendent les seuils comparables :
      * la base est la même pour tous les seuils : perte nette de TOUTES les annulations
        sans dépôt (revenu x (1 - taux de revente de la tranche)). Le seuil ne change
        que ce qui est récupéré, pas la perte de départ ;
      * l'acompte n'est demandé qu'aux réservations SANS dépôt : une réservation
        « Non Refund » est déjà payée, une « Refundable » a déjà versé un dépôt.

    Pour chaque politique :
      * acompte conservé : seulement pour les annulations signalées, et seulement si
        l'annulation tombe après la date limite de remboursement
      * perte nette restante = perte nette totale - acompte conservé
      * clients fidèles exposés = réservations signalées qui ne sont PAS annulées
      * taux de fuite maximal acceptable = acompte conservé / revenu des
        fidèles exposés : si une plus grande part des fidèles renonce à
        réserver, l'acompte coûte plus qu'il ne rapporte

    L'acompte est conservé même si la chambre est revendue : la part couverte
    peut donc dépasser 100 % (l'hôtel gagne alors plus que ce qu'il perdait).
    Un taux de fuite maximal supérieur à 100 % signifie que l'acompte resterait
    rentable même si TOUS les fidèles exposés renonçaient.
    """
    policies = policies or DEPOSIT_POLICIES
    if "deposit_type" in bookings:
        sans_depot = (bookings["deposit_type"] == "No Deposit").to_numpy()
    else:
        sans_depot = np.ones(len(bookings), dtype=bool)

    # Base commune : perte nette de toutes les annulations sans dépôt
    tous = bookings.loc[sans_depot]
    annulees = tous["is_canceled"] == 1
    taux = tous["delay_category"].map(resale).astype(float).fillna(float(resale.mean()))
    perte_totale = float((tous["stay_revenue"] * (1 - taux)).where(annulees, 0.0).sum())

    # Réservations à qui on demande un acompte : signalées ET sans dépôt
    b = bookings.loc[np.asarray(flagged, dtype=bool) & sans_depot].copy()
    cancelled = b["is_canceled"] == 1
    loyal_revenue = b.loc[~cancelled, "stay_revenue"].sum()

    rows = []
    for name, (kind, value, refundable_until) in policies.items():
        amount = deposit_amount(b["adr"], b["nights"], kind, value)
        kept = cancelled.copy()
        if refundable_until is not None:
            # Remboursé si l'annulation a lieu tôt : conservé seulement en cas d'annulation tardive
            kept &= b["cancel_delay"] < refundable_until
        recovered = float(amount[kept.to_numpy()].sum())
        rows.append(
            {
                "politique": name,
                "reservations_signalees": len(b),
                "annulations_signalees": int(cancelled.sum()),
                "perte_nette_totale": perte_totale,
                "acompte_conserve": recovered,
                "perte_nette_restante": perte_totale - recovered,
                "part_perte_couverte": recovered / perte_totale if perte_totale else 0.0,
                "fideles_exposes": int((~cancelled).sum()),
                "taux_fuite_max_acceptable": recovered / loyal_revenue if loyal_revenue else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("politique")


# ===========================================================================
# 4. Seuils de décision F-beta et action graduée
# ===========================================================================
def fbeta_by_threshold(y_true, y_proba, betas=(0.5, 1, 2)) -> pd.DataFrame:
    """Précision, rappel et F-beta pour chaque seuil possible."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision et recall ont un élément de plus que thresholds : on retire le dernier
    table = pd.DataFrame({"seuil": thresholds, "precision": precision[:-1], "rappel": recall[:-1]})
    for beta in betas:
        b2 = beta**2
        denom = b2 * table["precision"] + table["rappel"]
        # where : évite la division par zéro quand précision et rappel sont nuls
        table[f"f{beta:g}"] = np.where(
            denom > 0, (1 + b2) * table["precision"] * table["rappel"] / denom.where(denom > 0, 1), 0.0
        )
    return table


def best_threshold(y_true, y_proba, beta: float) -> float:
    """Seuil qui maximise le F-beta (beta > 1 : rappel prioritaire ; beta < 1 : précision)."""
    table = fbeta_by_threshold(y_true, y_proba, betas=(beta,))
    meilleur = int(table[f"f{beta:g}"].to_numpy().argmax())  # position du F-beta maximal
    return float(table["seuil"].iloc[meilleur])


def threshold_summary(y_true, y_proba, thresholds) -> pd.DataFrame:
    """Matrice de confusion résumée pour quelques seuils (lecture métier)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    rows = []
    for t in thresholds:
        alert = y_proba >= t                       # réservations signalées
        tp = int((alert & (y_true == 1)).sum())    # annulations détectées
        fp = int((alert & (y_true == 0)).sum())    # fidèles signalés à tort
        fn = int((~alert & (y_true == 1)).sum())   # annulations ratées
        rows.append(
            {
                "seuil": round(float(t), 3),
                "alertes": int(alert.sum()),
                "vraies_annulations": tp,
                "fideles_signales": fp,
                "annulations_ratees": fn,
                "precision": tp / max(tp + fp, 1),
                "rappel": tp / max(tp + fn, 1),
            }
        )
    return pd.DataFrame(rows)


ACTIONS = {
    "Faible": ("Aucune action", "Confirmation automatique standard."),
    "Modéré": (
        "Relance sans engagement",
        "Rappel par courriel et SMS avec un lien pour confirmer ou annuler en un clic, "
        "sans acompte : une annulation anticipée se revend mieux.",
    ),
    "Élevé": (
        "Acompte et surréservation",
        "Acompte demandé (politique choisie par l'hôtel) et chambre comptée dans la marge "
        "de surréservation de la date d'arrivée.",
    ),
}


def risk_level(p: float, low: float, high: float) -> str:
    """Zone de risque : faible sous le seuil F2, élevé au-dessus du seuil F0,5."""
    if p >= high:
        return "Élevé"
    if p >= low:
        return "Modéré"
    return "Faible"


# Réservation non remboursable : le client a payé la totalité du séjour.
# S'il annule, l'hôtel garde le paiement : il ne perd rien, et la revente
# éventuelle de la chambre est un gain supplémentaire.
ACTION_NON_REMBOURSABLE = (
    "Séjour déjà payé",
    "Aucune relance ni acompte : le client a payé la totalité du séjour et l'hôtel garde ce "
    "paiement en cas d'annulation. La chambre peut entrer dans la marge de surréservation : "
    "si elle est libérée, la revendre est un gain.",
)
# Réservation remboursable : un dépôt existe déjà, on ne redemande pas d'acompte.
ACTION_REMBOURSABLE_ELEVE = (
    "Surréservation (dépôt déjà versé)",
    "Pas de nouvel acompte : un dépôt remboursable a déjà été versé. La chambre est comptée "
    "dans la marge de surréservation de la date d'arrivée.",
)


def action_for(level: str, deposit_type: str | None = None) -> tuple[str, str]:
    """Action recommandée selon la zone de risque ET le type de dépôt."""
    if deposit_type == "Non Refund":
        return ACTION_NON_REMBOURSABLE
    if deposit_type == "Refundable" and level == "Élevé":
        return ACTION_REMBOURSABLE_ELEVE
    return ACTIONS[level]


def expected_loss(proba, revenue, deposit_type=None) -> np.ndarray:
    """Perte espérée = probabilité x revenu, nulle pour une réservation non remboursable.

    Une réservation « Non Refund » est payée d'avance : l'hôtel conserve le paiement en cas
    d'annulation, sa perte est donc de 0 €. Un dépôt « Refundable » est rendu au client :
    la perte reste probabilité x revenu.
    """
    proba = np.asarray(proba, dtype=float)
    loss = proba * np.asarray(revenue, dtype=float)
    if deposit_type is not None:
        paid = np.asarray(deposit_type, dtype=object) == "Non Refund"
        loss = np.where(paid, 0.0, loss)
    return loss


def recommend_actions(proba, revenue, low: float, high: float, deposit_type=None) -> pd.DataFrame:
    """Zone de risque, action recommandée et perte espérée pour chaque réservation.

    deposit_type (facultatif) : type de dépôt de chaque réservation. Une réservation non
    remboursable a une perte espérée nulle et ne reçoit ni relance ni acompte.
    """
    proba = np.asarray(proba, dtype=float)
    revenue = np.asarray(revenue, dtype=float)
    deposits = [None] * len(proba) if deposit_type is None else list(deposit_type)
    levels = [risk_level(p, low, high) for p in proba]
    return pd.DataFrame(
        {
            "proba_annulation": proba,
            "revenu_en_jeu": revenue,
            "perte_esperee": expected_loss(proba, revenue, deposit_type),  # sert à prioriser
            "niveau_risque": levels,
            "action_recommandee": [action_for(lvl, dep)[0] for lvl, dep in zip(levels, deposits)],
        }
    )


# ===========================================================================
# 5. Surréservation par niveau de service
# ===========================================================================
def show_distribution(p_cancel) -> np.ndarray:
    """Loi exacte du nombre de clients présents (loi de Poisson binomiale).

    Chaque réservation i se présente avec la probabilité (1 - p_i). On ajoute
    les réservations une par une : dist[k] = P(exactement k présents).
    """
    dist = np.array([1.0])
    for p in np.asarray(p_cancel, dtype=float):
        new = np.zeros(len(dist) + 1)
        new[:-1] += dist * p          # annulation : le nombre de présents ne change pas
        new[1:] += dist * (1.0 - p)   # le client vient : un présent de plus
        dist = new
    return dist


def overbooking_plan(p_cancel_existing, capacity: int, p_cancel_new: float, max_extra: int = 40) -> pd.DataFrame:
    """Effet de k réservations acceptées au-delà de la capacité, pour k = 0 à max_extra."""
    p_cancel_existing = np.asarray(p_cancel_existing, dtype=float)
    rows = []
    for k in range(max_extra + 1):
        probs = np.concatenate([p_cancel_existing, np.full(k, p_cancel_new)])
        dist = show_distribution(probs)
        shows = np.arange(len(dist))                 # nombres de présents possibles
        occupied = np.minimum(shows, capacity)       # pas plus que la capacité
        walked = np.maximum(shows - capacity, 0)     # présents sans chambre
        exp_occ = float((dist * occupied).sum())
        rows.append(
            {
                "reservations_supplementaires": k,
                "presents_esperes": float((dist * shows).sum()),
                "chambres_occupees_esperees": exp_occ,
                "taux_occupation_espere": exp_occ / capacity if capacity else 0.0,
                "clients_reloges_esperes": float((dist * walked).sum()),
                "prob_au_moins_un_reloge": float(dist[shows > capacity].sum()),
            }
        )
    return pd.DataFrame(rows)


def max_safe_overbooking(plan: pd.DataFrame, max_walk_prob: float) -> pd.Series:
    """Plus grand k dont la probabilité de reloger au moins un client reste <= max_walk_prob."""
    safe = plan[plan["prob_au_moins_un_reloge"] <= max_walk_prob]
    return safe.iloc[-1] if len(safe) else plan.iloc[0]
