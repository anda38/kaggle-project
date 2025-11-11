# === Importation des librairies nécessaires ===
import pandas as pd
import numpy as np
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # Ignorer les avertissements pour plus de clarté
# === Chargement des données ===
train = pd.read_csv("/Users/andalouse/data/train.csv")
test = pd.read_csv("/Users/andalouse/data/test.csv")
census = pd.read_csv("/Users/andalouse/data/census_starter.csv")
revealed = pd.read_csv("/Users/andalouse/data/revealed_test.csv")

# === Garder seulement les colonnes pertinentes ===
train = train[["row_id", "cfips", "first_day_of_month", "microbusiness_density"]]  # Garde les colonnes utiles
train = train.sort_values(["cfips", "first_day_of_month"])  # Trie les données par comté et par date

# === Calcul du taux de croissance (variable cible) ===
train["growth_rate"] = train.groupby("cfips")["microbusiness_density"].pct_change()  # Calcul du pourcentage de changement mois à mois
train["growth_rate"] = train["growth_rate"].replace([np.inf, -np.inf], np.nan)        # Supprime les infinis éventuels
train["growth_rate"] = train["growth_rate"].clip(-0.8, 0.8)                           # Limite les valeurs extrêmes
train = train.dropna(subset=["growth_rate"])                                          # Supprime les lignes sans cible

# === Création des variables de décalage (lags) ===
for lag in [1, 2, 3]:
    train[f"mbd_lag_{lag}"] = train.groupby("cfips")["microbusiness_density"].shift(lag)  # Valeur des 1, 2, 3 mois précédents

# === Création des moyennes glissantes ===
for window in [3, 6]:
    train[f"roll_mean_{window}"] = (
        train.groupby("cfips")["microbusiness_density"]
        .transform(lambda x: x.shift(1).rolling(window).mean())  # Moyenne des valeurs précédentes sur 3 et 6 mois
    )

# === Extraction de la date en variables année et mois ===
train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])  # Conversion en format date
train["year"] = train["first_day_of_month"].dt.year   # Extraction de l’année
train["month"] = train["first_day_of_month"].dt.month # Extraction du mois

# === Préparation des variables de recensement (census) ===
census = census.rename(columns={  # Renomme les colonnes de 2020 pour simplifier les noms
    "median_hh_inc_2020": "median_hh_inc",
    "pct_bb_2020": "pct_bb",
    "pct_college_2020": "pct_college",
    "pct_foreign_born_2020": "pct_foreign_born",
    "pct_it_workers_2020": "pct_it_workers"
})
cols_to_keep = ["cfips", "median_hh_inc", "pct_bb", "pct_college", "pct_foreign_born", "pct_it_workers"]  # Liste des colonnes utiles
train = train.merge(census[cols_to_keep], on="cfips", how="left")  # Fusion des données socio-économiques sur le code du comté

# === Découpage temporel du jeu de données (train / validation) ===
cutoff = pd.Timestamp("2022-08-01")  # Date de coupure temporelle
train_df = train[train["first_day_of_month"] <= cutoff]  # Données d'entraînement (avant août 2022)
val_df = train[train["first_day_of_month"] > cutoff]     # Données de validation (après août 2022)

# === Définition de la liste des variables explicatives (features) ===
features = [
    "cfips", "month", "year",
    "mbd_lag_1", "mbd_lag_2", "mbd_lag_3",
    "roll_mean_3", "roll_mean_6",
    "median_hh_inc", "pct_bb", "pct_college", "pct_foreign_born", "pct_it_workers"
]

# === Définition de la fonction de métrique SMAPE ===
def smape(y_true, y_pred):
    num = np.abs(y_true - y_pred)                           # Valeurs absolues des erreurs
    den = (np.abs(y_true) + np.abs(y_pred)) / 2             # Moyenne des valeurs vraies et prédites
    mask = (y_true != 0) | (y_pred != 0)                    # Évite la division par zéro
    return 100 * np.mean(num[mask] / den[mask])             # Retourne le SMAPE en pourcentage


# === Entraînement du modèle LightGBM ===
# === Entraînement du modèle LightGBM ===
model = LGBMRegressor(               # Initialisation du modèle LightGBM
    objective="regression_l1",       # Fonction de perte robuste aux valeurs extrêmes (L1)
    learning_rate=0.05,              # Taux d’apprentissage
    n_estimators=700,                # Nombre total d’arbres
    num_leaves=64,                   # Complexité des arbres (nombre de feuilles)
    subsample=0.9,                   # Proportion d’échantillons utilisés par arbre
    colsample_bytree=0.8,            # Proportion de variables utilisées par arbre
    random_state=42,                 # Graine aléatoire pour reproductibilité
)

model.fit(train_df[features], train_df["growth_rate"])  # Entraîne le modèle sur les variables d’entrée et la cible

# === Validation du modèle ===
val_pred_growth = model.predict(val_df[features])            # Prédictions du taux de croissance sur le jeu de validation
val_pred_density = (1 + val_pred_growth) * val_df["mbd_lag_1"]  # Conversion du taux de croissance en densité réelle
smape_val = smape(val_df["microbusiness_density"], val_pred_density)  # Calcul du SMAPE sur la validation
mae_val = mean_absolute_error(val_df["microbusiness_density"], val_pred_density)  # Calcul du MAE
# Les deux métriques évaluent la précision du modèle

# === Préparation des données de test ===
test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])  # Conversion du champ date
test["year"] = test["first_day_of_month"].dt.year                        # Extraction de l’année
test["month"] = test["first_day_of_month"].dt.month                      # Extraction du mois

# === Initialisation de l’historique pour les prévisions récursives ===
hist = train.groupby("cfips").tail(6)[["cfips", "first_day_of_month", "microbusiness_density"]].copy()  # Six derniers mois par comté
hist = hist.sort_values(["cfips", "first_day_of_month"])  # Trie chronologiquement l’historique
results = []  # Liste pour stocker les résultats de prédiction

# === Boucle de prévision récursive pour chaque mois futur ===
for date in sorted(test["first_day_of_month"].unique()):  # Parcourt chaque mois à prédire
    print(f"⏩ Predicting {date.date()}")

    temp = hist.groupby("cfips").tail(6).copy()  # Récupère les six dernières observations par comté

    # Création des variables de décalage (lags)
    for lag in [1, 2, 3]:
        temp[f"mbd_lag_{lag}"] = temp.groupby("cfips")["microbusiness_density"].shift(lag)

    # Création des moyennes mobiles
    for window in [3, 6]:
        temp[f"roll_mean_{window}"] = temp.groupby("cfips")["microbusiness_density"].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

    # Garde la dernière ligne complète par comté
    latest = temp.dropna(subset=["mbd_lag_1"]).drop_duplicates(subset=["cfips"], keep="last")
    latest = latest[["cfips", "mbd_lag_1", "mbd_lag_2", "mbd_lag_3", "roll_mean_3", "roll_mean_6"]]

    # Fusionne avec les données de test et de recensement
    step_df = test[test["first_day_of_month"] == date].merge(latest, on="cfips", how="left")
    step_df = step_df.merge(census[cols_to_keep], on="cfips", how="left")
    step_df[features] = step_df[features].ffill().bfill()  # Remplit les valeurs manquantes par propagation

    # Prédiction du modèle sur le mois courant
    growth_pred = model.predict(step_df[features])          # Prédiction du taux de croissance
    step_df["pred_density"] = (1 + growth_pred) * step_df["mbd_lag_1"]  # Conversion en densité prédite

    # Mise à jour de l’historique avec les nouvelles prédictions
    hist = pd.concat([
        hist,
        step_df[["cfips", "first_day_of_month", "pred_density"]]
        .rename(columns={"pred_density": "microbusiness_density"})
    ])

    results.append(step_df[["row_id", "pred_density"]])  # Sauvegarde des prédictions du mois

# === Création du fichier final de soumission ===
submission = pd.concat(results).rename(columns={"pred_density": "microbusiness_density"})  # Concatène tous les mois
submission.to_csv("submission.csv", index=False)  # Sauvegarde le fichier CSV


print("✅ Forecast complete. Saved as submission.csv.")
print(submission.head())

# === Évaluation finale avec le jeu de test révélé ===
merged = submission.merge(  # Fusion des prédictions et des vraies valeurs du jeu révélé
    revealed[["row_id", "microbusiness_density"]],
    on="row_id", how="inner", suffixes=("_pred", "_true")
)

mae = np.mean(np.abs(merged.microbusiness_density_pred - merged.microbusiness_density_true))  # Calcul du MAE global
smape_val = smape(merged.microbusiness_density_true, merged.microbusiness_density_pred)       # Calcul du SMAPE global

# === Section de visualisation pour le rapport ===
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Style général pour des graphiques homogènes et lisibles
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk", font_scale=1)
sns.set_palette("muted")

# === 1️⃣ Graphique temporel d’un comté (Time-Series Forecast) ===
cfips_example = 1001  # Exemple : comté 1001
train_cfips = train[train.cfips == cfips_example].copy()  # Données d’entraînement pour ce comté
pred_cfips = submission[submission.row_id.str.startswith(str(cfips_example))].copy()  # Prédictions correspondantes
pred_cfips["first_day_of_month"] = pd.to_datetime(pred_cfips["row_id"].str[-10:])  # Extraction de la date depuis row_id

# Combine les données historiques et prédites pour un affichage continu
all_cfips = pd.concat([
    train_cfips[["first_day_of_month", "microbusiness_density"]].assign(Source="Train"),
    pred_cfips[["first_day_of_month", "microbusiness_density"]].assign(Source="Forecast")
])

plt.figure(figsize=(11, 6))
sns.lineplot(  # Courbes lissées pour visualiser l’évolution temporelle
    data=all_cfips,
    x="first_day_of_month",
    y="microbusiness_density",
    hue="Source",
    style="Source",
    markers=True,
    dashes=False,
    linewidth=2.5,
    palette={"Train": "#1f77b4", "Forecast": "#ff7f0e"},
)

# Zone ombrée indiquant la période de prévision
forecast_start = pred_cfips["first_day_of_month"].min()
plt.axvspan(forecast_start, all_cfips["first_day_of_month"].max(),
            color="#ff7f0e", alpha=0.08, label="Période de prévision")

plt.title(f"📈 Prévision de la densité des microentreprises — comté {cfips_example}",
          fontsize=17, fontweight="bold", pad=15)
plt.xlabel("Date", fontsize=13)
plt.ylabel("Densité des microentreprises", fontsize=13)
plt.legend(frameon=True, loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# === 2️⃣ Graphique d’importance des variables (Feature Importance) ===
imp = pd.DataFrame({  # Crée une table des importances
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)  # Trie les variables de la moins à la plus importante

plt.figure(figsize=(8, 6))
sns.barplot(  # Graphique en barres horizontales
    data=imp,
    y="Feature",
    x="Importance",
    palette="Blues_d"
)
plt.title("🔍 Importance des variables (Modèle LightGBM)",
          fontsize=17, fontweight="bold", pad=12)
plt.xlabel("Score d’importance", fontsize=13)
plt.ylabel("")
plt.tight_layout()
plt.show()


# === 3️⃣ Nuage de points de validation (Predicted vs True) ===
val_results = pd.DataFrame({  # Assemble les valeurs réelles et prédites
    "True": val_df["microbusiness_density"],
    "Predicted": val_pred_density
})

plt.figure(figsize=(7, 7))
sns.scatterplot(  # Représente la corrélation entre valeurs réelles et prédites
    data=val_results,
    x="True",
    y="Predicted",
    alpha=0.6,
    s=50,
    color="#4C72B0",
    edgecolor="white",
    linewidth=0.5
)

# Ajout de la ligne rouge idéale (y = x)
max_val = max(val_results.max())
plt.plot([0, max_val], [0, max_val], "r--", lw=2.5, label="Prédiction parfaite")

plt.title("🎯 Validation — Valeurs prédites vs réelles",
          fontsize=17, fontweight="bold", pad=15)
plt.xlabel("Valeurs réelles", fontsize=13)
plt.ylabel("Valeurs prédites", fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
