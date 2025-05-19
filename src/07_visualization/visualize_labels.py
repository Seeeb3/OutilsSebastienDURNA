import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

# Charger les fichiers
auto = pd.read_csv("data/processed/death_row_labeled.csv")
zero = pd.read_csv("data/processed/death_row_zero_shot.csv")

# Nettoyer les colonnes (les convertir de string à list)
auto["auto_labels"] = auto["auto_labels"].apply(eval)
zero["zero_shot_labels"] = zero["zero_shot_labels"].apply(eval)

# Binarisation des labels
mlb = MultiLabelBinarizer()
auto_bin = mlb.fit_transform(auto["auto_labels"])
zero_bin = mlb.transform(zero["zero_shot_labels"])

# Fréquence des labels
auto_freq = pd.Series([label for sublist in auto["auto_labels"] for label in sublist]).value_counts()
zero_freq = pd.Series([label for sublist in zero["zero_shot_labels"] for label in sublist]).value_counts()

# Affichage comparatif
freq_df = pd.DataFrame({
    "Auto-labels": auto_freq,
    "Zero-shot labels": zero_freq
}).fillna(0)

freq_df.plot(kind="bar", figsize=(10, 5), title="Comparaison des fréquences des labels")
plt.tight_layout()
plt.savefig("data/visualization/label_frequency_comparison.png")
plt.close()