import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score, classification_report, hamming_loss

# Charger les fichiers
auto = pd.read_csv("data/processed/death_row_labeled.csv")
zero = pd.read_csv("data/processed/death_row_zero_shot.csv")

# Nettoyage et alignement
auto = auto.dropna(subset=["auto_labels"])
zero = zero[zero["execution_id"].isin(auto["execution_id"])]

# Conversion des labels (string -> list)
auto["auto_labels"] = auto["auto_labels"].apply(eval)
zero["zero_shot_labels"] = zero["zero_shot_labels"].apply(eval)

# Binarisation
mlb = MultiLabelBinarizer()
auto_bin = mlb.fit_transform(auto["auto_labels"])
zero_bin = mlb.transform(zero["zero_shot_labels"])

# Supprimer la classe "none" si présente
mlb_classes = list(mlb.classes_)
if "none" in mlb_classes:
    none_index = mlb_classes.index("none")
    auto_bin = pd.DataFrame(auto_bin).drop(columns=none_index).values
    zero_bin = pd.DataFrame(zero_bin).drop(columns=none_index).values
    mlb_classes.remove("none")

# Jaccard moyen
jaccard = jaccard_score(auto_bin, zero_bin, average="samples")
print(f"Jaccard moyen : {jaccard:.3f}")

# Classification report
print("\nClassification Report (micro/macro):")
print(classification_report(auto_bin, zero_bin, target_names=mlb_classes))

# Hamming Loss
h_loss = hamming_loss(auto_bin, zero_bin)
print(f"Hamming Loss : {h_loss:.3f}")

# Exact Match Ratio
exact_match = (auto_bin == zero_bin).all(axis=1).mean()
print(f"Exact Match Ratio : {exact_match:.3f}")