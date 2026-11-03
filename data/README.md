# Données

## Source
**Hotel Booking Demand**, publié par Antonio, Almeida et Nunes (2019, *Data in Brief*) et repris par le projet TidyTuesday (semaine du 2020-02-11).

- Fiche et dictionnaire des variables : <https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md>
- Fichier : <https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv>

## Contenu
119 390 réservations de deux hôtels portugais (un hôtel urbain, un hôtel de villégiature), arrivées entre juillet 2015 et août 2017, 32 colonnes. Cible : `is_canceled` (1 = annulée).

## Emplacement
Le fichier `hotels.csv` est attendu dans `data/raw/`. Il n'est pas versionné (16 Mo) : s'il est absent, `src/data_loader.load_raw_data()` le télécharge automatiquement depuis le dépôt TidyTuesday.
