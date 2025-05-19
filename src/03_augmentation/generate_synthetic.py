import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# Charger le modèle de paraphrase BART
model_name = "eugenesiow/bart-paraphrase"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Créer le dossier de sortie si nécessaire
os.makedirs("data/synthetic", exist_ok=True)

# Charger les données existantes
df = pd.read_csv("data/processed/death_row_clean.csv")

# Sélectionner un sous-ensemble pour générer des paraphrases
sampled = df.sample(n=50, random_state=42).copy()

def generate_paraphrase(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = text
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    generated_ids = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    paraphrased = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return paraphrased


# Générer les paraphrases avec barre de progression
tqdm.pandas()
sampled["last_statement"] = sampled["last_statement"].astype(str).progress_apply(generate_paraphrase)
sampled["synthetic"] = True

# Ajouter une colonne "synthetic" aux originaux
df["synthetic"] = False

# Fusionner et enregistrer
augmented_df = pd.concat([df, sampled], ignore_index=True)
augmented_df.to_csv("data/synthetic/death_row_augmented.csv", index=False)

print("Paraphrases générées avec T5 et enregistrées dans data/synthetic/death_row_augmented.csv")