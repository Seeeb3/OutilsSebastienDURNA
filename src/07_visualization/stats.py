import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import geopandas as gpd
from unidecode import unidecode

# Charger les données nettoyées
df = pd.read_csv("data/processed/death_row_clean.csv")

# Histogramme des exécutions par année
df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
plt.figure(figsize=(10, 5))
sns.countplot(x="year", data=df, order=sorted(df["year"].dropna().unique()))
plt.xticks(rotation=45)
plt.title("Nombre d'exécutions par année")
plt.tight_layout()
plt.savefig("data/visualization/executions_per_year.png")
plt.close()

# Répartition ethnique des condamnés à mort
plt.figure(figsize=(6, 6))
df["race"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Répartition ethnique des condamnés à mort")
plt.ylabel("")
plt.tight_layout()
plt.savefig("data/visualization/race_distribution.png")
plt.close()

# Nuage de mots des déclarations
text = " ".join(df["last_statement"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Nuage de mots des last statements")
plt.tight_layout()
plt.savefig("data/visualization/wordcloud_last_statements.png")
plt.close()

# Longueur moyenne des déclarations
df["statement_length"] = df["last_statement"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8, 4))
sns.histplot(df["statement_length"], bins=30, kde=True)
plt.title("Distribution de la longueur des déclarations")
plt.xlabel("Nombre de mots")
plt.tight_layout()
plt.savefig("data/visualization/statement_length_distribution.png")
plt.close()

# Repartition des condamnés par comté
plt.figure(figsize=(12, 6))
df["county"].value_counts().head(10).plot.bar()
plt.title("Top 10 des comtés avec le plus de condamnés à mort")
plt.ylabel("Nombre de condamnés")
plt.xlabel("Comté")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/visualization/top_counties.png")
plt.close()

# Repartition des condamnés par âge (date d'exécution)
plt.figure(figsize=(10, 5))
max_age = int(df["age"].max()) + 1
bins = list(range(0, max_age + 2, 2))
sns.histplot(df["age"], bins=bins, kde=False)
plt.title("Distribution de l'âge des condamnés à mort")
plt.xlabel("Âge")
plt.ylabel("Nombre de condamnés")
plt.tight_layout()
plt.savefig("data/visualization/age_distribution.png")
plt.close()

print("Graphiques générés dans data/visualization/")

# Carte des exécutions par comté au Texas
texas_geojson_path = "data/raw/texas_counties.geojson"
if os.path.exists(texas_geojson_path):
    # Préparer les données
    county_counts = df["county"].value_counts().reset_index()
    county_counts.columns = ["county", "executions"]
    county_counts["county"] = county_counts["county"].str.upper().apply(unidecode)

    # Charger la carte des comtés du Texas
    gdf = gpd.read_file(texas_geojson_path)
    gdf["COUNTY_NAME"] = gdf["COUNTY"].str.replace(" County", "", regex=False).str.upper().apply(unidecode)

    # Fusion
    merged = gdf.merge(county_counts, left_on="COUNTY_NAME", right_on="county", how="left")
    merged["executions"] = merged["executions"].fillna(0)

    # Tracer la carte avec plus de lisibilité
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    merged.plot(column="executions", ax=ax, legend=True, cmap="OrRd", edgecolor="grey", linewidth=0.5)

    # Annoter les comtés principaux
    top = merged.sort_values("executions", ascending=False).head(5)
    for _, row in top.iterrows():
        centroid = row["geometry"].centroid
        ax.annotate(
            row["COUNTY_NAME"].title(),
            xy=(centroid.x, centroid.y),
            ha="center",
            fontsize=9,
            color="black"
        )

    ax.set_title("Exécutions par comté au Texas (1976–2024)", fontsize=14, pad=15)
    ax.axis("off")
    plt.figtext(0.5, 0.01, "Source : TDCJ Death Row — https://www.tdcj.texas.gov/", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("data/visualization/executions_by_county_map.png")
    plt.close()
else:
    print(f"❌ Fichier GeoJSON non trouvé : {texas_geojson_path}")
