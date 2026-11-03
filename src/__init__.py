"""Pipeline de prédiction des annulations de réservations hôtelières.

Modules
-------
config            : chemins, colonnes, paramètres de reproductibilité
data_loader       : chargement, audit de qualité, nettoyage, séparation 80/20
transformers      : transformateurs personnalisés (TransformerMixin)
preprocessing     : pipelines par type de variable et ColumnTransformer
feature_selection : sélection des caractéristiques (SelectFromModel)
model             : pipeline complet, comparaison, optimisation, sauvegarde
evaluation        : métriques, graphiques, importance par permutation
prescriptive      : seuil optimal, actions recommandées, surréservation
train             : script d'entraînement en ligne de commande
"""
