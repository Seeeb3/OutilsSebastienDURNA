import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Charger le classifieur zero-shot
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Labels candidats à détecter
candidate_labels = ["regret", "faith", "family", "justice", "peace", "forgiveness"]

# Charger les données
df = pd.read_csv("data/processed/death_row_clean.csv")
df["last_statement"] = df["last_statement"].fillna("")

# Appliquer le classifieur à chaque phrase
tqdm.pandas()
def classify(text):
    output = classifier(text, candidate_labels=candidate_labels, multi_label=True)
    top_labels = output["labels"][:3]
    return top_labels

df["zero_shot_labels"] = df["last_statement"].progress_apply(classify)

# Sauvegarder le résultat
df.to_csv("data/processed/death_row_zero_shot.csv", index=False)
print("Prédictions zéro-shot sauvegardées dans data/processed/death_row_zero_shot.csv")

