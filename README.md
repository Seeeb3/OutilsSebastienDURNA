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
