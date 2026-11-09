# Prédiction des annulations de réservations hôtelières

Pipeline scikit-learn complet (prétraitement, transformateurs personnalisés, sélection des caractéristiques, classification) pour prédire l'annulation d'une réservation. Il est prolongé par une **analyse prescriptive**, en notebook et dans une application Streamlit, qui chiffre le coût réel des annulations et ce qu'un acompte permettrait de récupérer.

> Évaluation sommative 2, *Intelligence artificielle appliquée au prétraitement des données* (030513 IFM), Collège La Cité, automne 2026.
> Auteur : Yacine Lezoul

---

## Résultats clés

**Modèle** : HistGradientBoosting avec pondération des classes, probabilités calibrées. Sur l'ensemble de test (17 197 réservations jamais vues) :

| Seuil de décision | Rôle | Rappel | Précision |
|---|---|---|---|
| **F2** (0,17) | détecter : relance sans engagement | **93 %** | 51 % |
| **F0,5** (0,54) | être sûr : acompte, surréservation | 59 % | **75 %** |

Les deux seuils sont choisis par validation croisée **dans l'entraînement**. Autres résultats : ROC-AUC 0,895, F2 maximal 0,798.

**Analyse prescriptive** (juillet 2015 à août 2017) :

| Question | Réponse |
|---|---|
| Combien coûtent les annulations sans dépôt ? | 13,1 M€ brut, mais **3,0 M€ net** après revente des chambres (11,7 % du revenu réalisé) |
| Quand ont-elles lieu ? | 46 % dans le dernier mois avant l'arrivée, qui concentre 53 % de la perte nette |
| Quelle part des chambres est revendue ? | de 61 % (annulation dans la dernière semaine) à 86 % (plus de 3 mois avant) |
| Que rapporterait un acompte ? | au seuil F0,5, de 72 % à 232 % de la perte nette selon la politique |

---

## Structure du dépôt

```
hotel-cancellation-pipeline/
│
├── data/
│   ├── raw/
│   │   └── hotels.csv                  # données brutes (non versionnées, téléchargées automatiquement)
│   └── README.md                       # source et description des données
│
├── src/                                # scripts importés par les notebooks et l'application
│   ├── __init__.py
│   ├── config.py                       # chemins, colonnes de fuite, colonnes d'entrée, graine aléatoire
│   ├── data_loader.py                  # chargement, diagnostic de qualité, nettoyage, séparation 80/20
│   ├── transformers.py                 # transformateurs personnalisés (TransformerMixin) :
│   │                                   #   HotelFeatureEngineer, RareCategoryGrouper, OutlierCapper
│   ├── pipelines.py                    # TOUS LES PIPELINES, écrits sur le modèle du cours :
│   │                                   #   pipeline_continu, pipeline_discret, pipeline_binaire, pipeline_cat
│   │                                   #   preprocessor = ColumnTransformer([...])
│   │                                   #   pipeline_pretraitement = Pipeline([HotelFeatureEngineer, preprocessor])
│   │                                   #   selecteur = SelectFromModel(RandomForest)
│   │                                   #   model_pipeline = Pipeline([preprocessing, selection, model])
│   ├── evaluation.py                   # métriques, scorer F2, graphiques, bilan de la sélection
│   ├── prescriptive.py                 # délais d'annulation, taux de revente, manque à gagner,
│   │                                   #   scénarios d'acompte, seuils F2 / F0,5, surréservation
│   └── sauvegarde.py                   # enregistrer et recharger le pipeline entraîné
│
├── notebooks/
│   ├── 01_pipeline.ipynb               # ÉVALUATION : Parties 1 à 5 de la consigne
│   └── 02_analyse_prescriptive.ipynb   # ANALYSE D'IMPACT : pertes, revente, acomptes, surréservation
│
├── app/
│   └── app.py                          # application Streamlit (prédiction et prescriptif)
│
├── models/                             # produits par 01_pipeline.ipynb (versionnés, 9 Mo)
│   ├── pipeline_annulation.joblib      #   pipeline complet entraîné
│   ├── metadata.json                   #   seuils F2 / F0,5, métriques, date
│   └── test_set.csv                    #   ensemble de test
│
├── reports/
│   ├── figures/                        # graphiques exportés par les notebooks
│   └── presentation_annulations.pptx   # diapositives de présentation (17, avec notes)
│
├── docs/
│   ├── document_technique.docx         # document technique (Word, source)
│   └── document_technique.pdf          # document technique (version PDF)
│
├── tests/
│   └── test_transformers.py            # tests : transformateurs, pipelines, prescriptif
│
├── requirements.txt                    # versions fixées
├── .gitignore
└── README.md
```

Les notebooks sont numérotés dans l'ordre d'exécution : `02` utilise le pipeline enregistré par `01`.

**Livrables** : le notebook `notebooks/01_pipeline.ipynb` (Parties 1 à 5 de la consigne), les diapositives `reports/presentation_annulations.pptx` et le document technique `docs/document_technique.docx` (version PDF à côté).

---

## Le pipeline

Défini dans `src/pipelines.py`, sur le modèle du cours *Pipelines de prétraitement* :

```
model_pipeline = Pipeline([
    ("preprocessing", pipeline_pretraitement)
    │     ├── creation_variables : HotelFeatureEngineer            (transformateur personnalisé)
    │     └── colonnes : ColumnTransformer
    │           ├── num_continu : SimpleImputer → OutlierCapper → StandardScaler
    │           ├── num_discret : SimpleImputer → StandardScaler
    │           ├── binaire     : SimpleImputer
    │           └── cat         : SimpleImputer → RareCategoryGrouper → OneHotEncoder
    ├── ("selection", SelectFromModel(RandomForestClassifier))     47 colonnes sur 93
    └── ("model", HistGradientBoostingClassifier)                  calibré, seuil F2
])
```

**Règle suivie** : tout ce qui doit s'appliquer à une nouvelle réservation est fait **dans le pipeline** (valeurs manquantes, modalités « Undefined », valeurs extrêmes, encodage, nouvelles variables). Ce qui **supprime des lignes** de l'historique (33 241 doublons, réservations impossibles) est fait **une seule fois, avant** la séparation entraînement / test, car un pipeline ne peut pas retirer de lignes sans désaligner `X` et `y`.

Points de méthode :
- **Fuite de données évitée** : `reservation_status`, `reservation_status_date` et `assigned_room_type`, connues seulement après coup, sont exclues du modèle.
- **Comparaison équitable** sur le **F2 maximal** (rappel deux fois plus important que la précision, au meilleur seuil de chaque modèle) : régression logistique, forêt aléatoire et HistGradientBoosting, avec et sans `class_weight="balanced"`.
- **Calibration** (`CalibratedClassifierCV`) : la pondération des classes gonfle les probabilités, alors que l'analyse prescriptive les utilise directement.

---

## Installation

Prérequis : Python 3.11 ou plus récent (testé avec Python 3.13), Git.

```bash
git clone https://github.com/<votre-compte>/hotel-cancellation-pipeline.git
cd hotel-cancellation-pipeline

# Environnement virtuel
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Le jeu de données est téléchargé automatiquement au premier chargement. Pour l'ajouter à la main, placer `hotels.csv` dans `data/raw/`.

---

## Utilisation

| Étape | Commande | Durée |
|---|---|---|
| Application (directement) | `streamlit run app/app.py` | |
| 1. Pipeline et modèle | exécuter `notebooks/01_pipeline.ipynb` | environ 20 min |
| 2. Analyse prescriptive | exécuter `notebooks/02_analyse_prescriptive.ipynb` | environ 2 min |
| Tests | `python -m pytest -q` | quelques secondes |

Le pipeline entraîné est inclus dans `models/` : l'application et le notebook 02 fonctionnent dès le clonage. Réexécuter le notebook 01 réentraîne le modèle et remplace ces fichiers, avec des résultats identiques (graine aléatoire fixée).

### Application Streamlit

| Onglet | Contenu |
|---|---|
| Réservation | probabilité d'annulation, zone de risque, action recommandée |
| Lot de réservations | priorisation, import et export CSV |
| Pertes et acomptes | délais, taux de revente et manque à gagner ; simulateur d'acompte (montant, remboursement jusqu'à J-N, seuil) avec le taux de fuite maximal acceptable |
| Surréservation | réservations supplémentaires selon le risque de relogement accepté |

---

## Analyse prescriptive : principe

**Aucun chiffre inventé.** Le moment des annulations, le taux de revente (estimé nuit par nuit à partir de l'occupation réelle) et les pertes sont calculés à partir des données. Seuls le **niveau d'acompte** et le **risque de relogement accepté** sont des choix de l'hôtel, présentés comme des scénarios.

**Ce que les données ne permettent pas de mesurer** : combien de clients fidèles renonceraient à réserver devant un acompte, car ils n'apparaissent pas dans le fichier. On calcule à la place le **taux de fuite maximal acceptable**, au-delà duquel l'acompte coûte plus qu'il ne rapporte, et on propose un **test A/B** pour mesurer le taux réel.

---

## Limites

- Deux hôtels portugais entre 2015 et 2017 : il faut réentraîner le modèle avant de l'utiliser ailleurs.
- Séparation aléatoire : une validation temporelle (test sur 2017) serait plus réaliste.
- `required_car_parking_spaces` et `country` pourraient être complétées à l'arrivée du client (fuite partielle possible), à vérifier auprès de l'hôtel.
- La capacité des hôtels est estimée (99e centile de l'occupation observée), car elle n'est pas dans les données.

## Source des données

Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49. Version TidyTuesday : <https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11>
