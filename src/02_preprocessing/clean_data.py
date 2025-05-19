import pandas as pd
import numpy as np
import os

# Charger les données
df = pd.read_csv("data/raw/death_row_index.csv")

df["last_statement"] = df["last_statement"].where(df["last_statement"].notna(), None)
df["last_statement"] = df["last_statement"].str.strip().str.lower()
df["last_statement"] = df["last_statement"].replace([
    "no last statement given.",
    "no statement was made.",
    "spoken: no.",
    "none.",
    "no.",
    "no last statement given",
    "no statement given.",
    "this inmate declined to make a last statement.",
    "no last statement.",
    "none",
    "n/a",
    "no",
    ""
], np.nan)
df = df.dropna(subset=["last_statement"])

# Supprimer les colonnes entièrement vides
df = df.dropna(axis=1, how="all")
# Supprimer la colonne "TDCJ Number"
df = df.drop(columns=["TDCJ Number"], errors="ignore")

# Créer le dossier de sortie si besoin
os.makedirs("data/processed", exist_ok=True)

# Sauvegarder le fichier nettoyé
df.to_csv("data/processed/death_row_clean.csv", index=False)
print("Données nettoyées sauvegardées dans data/processed/death_row_clean.csv")
