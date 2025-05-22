# Zero-Shot Labelling on Last Statements

## Description

Projet réalisé dans le cadre du cours de Master 1 PluriTAL `Outils de Traitement de Corpus`.

Ce projet s'inscrit dans le cadre d'un travail de constitution et de traitement de corpus portant sur les **dernières déclarations des condamnés à mort au Texas**.

L’objectif est de constituer un corpus à partir des transcriptions officielles, d’analyser ce corpus linguistiquement, et d’appliquer des méthodes de labellisation automatique, notamment **l’apprentissage zero-shot**, afin d’annoter les propos tenus selon des thématiques récurrentes (regret, faith, family, justice, peace and forgiveness).

## Objectifs du projet

- Récupérer les dernières déclarations depuis le site officiel du Texas Department of Criminal Justice (TDCJ)
- Nettoyer et structurer les données
- Générer des données synthétiques via un modèle de paraphrase
- Annoter automatiquement les textes à l’aide de méthodes naïves et de zero-shot learning
- Comparer les résultats à l’aide de métriques (Jaccard, précision, rappel, F1, etc.)
- Produire des visualisations et des analyses interprétables

## Structure du projet

```
death_row_corpus/
├── data/
│   ├── raw/                    # Données brutes scrappées
│   ├── processed/              # Données nettoyées, enrichies, synthétiques
│   └── visualizations/         # Graphiques produits
│   └── synthetic/              # Dataset avec les données synthétiques
├── src/
│   ├── 01_scraping/            # Scripts de récupération des données
│   ├── 02_preprocessing/       # Scripts de nettoyage
│   ├── 03_augmentation/        # Génération de paraphrases (données synthétiques)
│   ├── 04_annotation/          # Labellisation zero-shot
│   ├── 05_transformer/         # Labellisation zero-shot
│   ├── 06_evaluation/          # Scripts d'évaluation des performances
│   └── 07_visualization/       # Scripts de visualisation statistique
├── requirements.txt            # Dépendances Python
└── README.md                   # Documentation du projet
```

## Technologies et bibliothèques

- `playwright` — Scraping web
- `pandas`, `matplotlib`, `seaborn` — Traitement de données & visualisation
- `transformers` (Hugging Face) — Modèle BART (`facebook/bart-large-mnli`) pour le zero-shot
- `scikit-learn` — Évaluation

## Livrables attendus

- Corpus structuré et enrichi
- Graphiques statistiques interprétables
- Données synthétiques générées
- Résultats de labellisation zero-shot
- Évaluation entre méthodes simples et modèles transformers

## Mesures d'évaluation

                precision    recall  f1-score   support

    faith           0.56      0.82      0.67       211
    family          0.70      0.97      0.81       296
    forgiveness     0.39      0.90      0.54        93
    justice         0.25      0.26      0.25        90
    peace           0.45      0.36      0.40       162
    regret          0.64      0.61      0.62       210

    micro avg       0.56      0.71      0.62      1062
    macro avg       0.50      0.65      0.55      1062
    weighted avg    0.56      0.71      0.61      1062
    samples avg     0.56      0.67      0.57      1062

Hamming Loss : 0.336

Exact Match Ratio : 0.055
