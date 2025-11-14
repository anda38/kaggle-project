# Import des libraries
import numpy as np
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")  # On ignore les warnings pour la lisibilité

# On charge les données
train = pd.read_csv("/Users/andalouse/data/train.csv")
test = pd.read_csv("/Users/andalouse/data/test.csv")
census = pd.read_csv("/Users/andalouse/data/census_starter.csv")
revealed = pd.read_csv("/Users/andalouse/data/revealed_test.csv")

# On drop les noms des counties et on garde que la clé cfips puis on trie dans le temps 
train = train[["row_id", "cfips", "first_day_of_month", "microbusiness_density"]]  # Garde les colonnes utiles
train = train.sort_values(["cfips", "first_day_of_month"])  # Trie les données par comté et par date

# On transforme la densité absolue en taux de croissance mensuel
train["growth_rate"] = train.groupby("cfips")["microbusiness_density"].pct_change()   # On calcule le pourcentage de changement mois à mois
train["growth_rate"] = train["growth_rate"].replace([np.inf, -np.inf], np.nan)        # On supprime les infinis éventuels
train["growth_rate"] = train["growth_rate"].clip(-0.8, 0.8)                           # On traite les valeurs extrêmes des deux côtés
train = train.dropna(subset=["growth_rate"])                                          # Et on supprime les lignes sans la target

# On crée les variables de retard 
for lag in [1, 2, 3]:
    train[f"mbd_lag_{lag}"] = train.groupby("cfips")["microbusiness_density"].shift(lag)  # Valeur des 1, 2, 3 mois précédents

# On crée des moyennes glissantes pour lisser les variations trop brusques
for window in [3, 6]:
    train[f"roll_mean_{window}"] = (
        train.groupby("cfips")["microbusiness_density"]
        .transform(lambda x: x.shift(1).rolling(window).mean())  # moyenne des valeurs précédentes sur 3 et 6 mois
    )

# On extrait l'année et le mois
train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])  # On convertit en format date
train["year"] = train["first_day_of_month"].dt.year   # extraction de l’année
train["month"] = train["first_day_of_month"].dt.month # extraction du mois

# On choisit liste les colonnes à garder pour le merge
cols_to_keep = [
    "cfips",
    "median_hh_inc_2020",
    "pct_bb_2020",
    "pct_college_2020",
    "pct_foreign_born_2020",
    "pct_it_workers_2020"
]
train = train.merge(census[cols_to_keep], on="cfips", how="left")  # On ... roulement de tambours ... merge

# On split la dataframe pour la partie apprentissage et la partie validation
cutoff = pd.Timestamp("2022-08-01")  # Date de coupure
train_df = train[train["first_day_of_month"] <= cutoff]  # Données d'entraînement (avant août 2022)
val_df = train[train["first_day_of_month"] > cutoff]     # Données de validation (après août 2022)

# On liste les features
features = [
    "cfips", "month", "year",
    "mbd_lag_1", "mbd_lag_2", "mbd_lag_3",
    "roll_mean_3", "roll_mean_6",
    "median_hh_inc_2020", "pct_bb_2020", "pct_college_2020", "pct_foreign_born_2020", "pct_it_workers_2020"
]

# On entraîne le modèle 
model = LGBMRegressor(               
    objective="regression_l1",       # On choisit L1 pour minimiser l’erreur absolue
    learning_rate=0.05,              # On choisit un taux d’apprentissage modéré
    n_estimators=700,                # On choisit un nombre d’arbres suffisant
    num_leaves=64,                   # On choisit un nombre de feuilles pour capturer la complexité
    subsample=0.9,                   # Proportion d’échantillons utilisés par arbre
    colsample_bytree=0.8,            # Proportion de variables utilisées par arbre
    random_state=42,                 # Random seed pour la reproductibilité
)

model.fit(train_df[features], train_df["growth_rate"])  # Entraîne le modèle sur les features et la target

# On stocke les prédictions et on reconvertit le taux de croissance en densité réelle
val_pred_growth = model.predict(val_df[features])            # Prédictions du taux de croissance sur le jeu de validation
val_pred_density = (1 + val_pred_growth) * val_df["mbd_lag_1"]  # Conversion du taux de croissance en densité réelle


# On définit le SMAPE
def smape(y_true, y_pred):
    num = np.abs(y_true - y_pred)                           
    den = (np.abs(y_true) + np.abs(y_pred)) / 2             
    mask = (y_true != 0) | (y_pred != 0)                    
    return 100 * np.mean(num[mask] / den[mask])             

smape_val = smape(val_df["microbusiness_density"], val_pred_density)  # Premier calcul du SMAPE sur la validation
mae_val = mean_absolute_error(val_df["microbusiness_density"], val_pred_density)  # Premier calcul du MAE

# On refait le même traitement datetime sur les données de test
test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])  
test["year"] = test["first_day_of_month"].dt.year                        
test["month"] = test["first_day_of_month"].dt.month                      

# On stocke les six derniers mois pour chaque county 
hist = train.groupby("cfips").tail(6)[["cfips", "first_day_of_month", "microbusiness_density"]].copy()  # Six derniers mois par county
hist = hist.sort_values(["cfips", "first_day_of_month"])  # On trie dans le temps
results = []  # On crée une liste pour stocker les résultats mensuels

# Boucle de prévision récursive mois par mois

for date in sorted(test["first_day_of_month"].unique()):  # Passe par chaque mois à prédire
    print(f"Prévision {date.date()}")

    temp = hist.groupby("cfips").tail(6).copy()  # On récupère les six dernières observations par comté

    # On crée les variables de retard pour les données test
    for lag in [1, 2, 3]:
        temp[f"mbd_lag_{lag}"] = temp.groupby("cfips")["microbusiness_density"].shift(lag)

    # Idem pour les moyennes glissantes
    for window in [3, 6]:
        temp[f"roll_mean_{window}"] = temp.groupby("cfips")["microbusiness_density"].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

    
    latest = temp.dropna(subset=["mbd_lag_1"]).drop_duplicates(subset=["cfips"], keep="last") # On supprime les lignes inutiles
    latest = latest[["cfips", "mbd_lag_1", "mbd_lag_2", "mbd_lag_3", "roll_mean_3", "roll_mean_6"]]

    # On merge les colonnes
    step_df = test[test["first_day_of_month"] == date].merge(latest, on="cfips", how="left")
    step_df = step_df.merge(census[cols_to_keep], on="cfips", how="left")
    step_df[features] = step_df[features].ffill().bfill()  # On remplit les valeurs manquantes par propagation
    

    # Prédiction du modèle sur le mois courant
    growth_pred = model.predict(step_df[features])          # Prédiction du taux de croissance
    step_df["pred_density"] = (1 + growth_pred) * step_df["mbd_lag_1"]  # On reconvertit en densité 

    # On met à jour de l’historique avec les nouvelles prédictions
    hist = pd.concat([
        hist,
        step_df[["cfips", "first_day_of_month", "pred_density"]]
        .rename(columns={"pred_density": "microbusiness_density"})
    ])

    results.append(step_df[["row_id", "pred_density"]])  # On sauvegarde les prédictions du mois

# On crée le fichier de soumission
submission = pd.concat(results).rename(columns={"pred_density": "microbusiness_density"}) 
submission.to_csv("submission.csv", index=False)  

print(submission.head())


# On évalue localement la performance avec le fichier revealed car la compétition est terminée le late submitting n’est plus possible
revealed = pd.read_csv("/Users/andalouse/data/revealed_test.csv")
submission = pd.read_csv("submission.csv")

merged = submission.merge(
    revealed[["row_id", "microbusiness_density"]],
    on="row_id",
    how="inner",
    suffixes=("_pred", "_true")
)

smape_val = smape(merged.microbusiness_density_true, merged.microbusiness_density_pred)
mae_val = np.mean(np.abs(merged.microbusiness_density_true - merged.microbusiness_density_pred))

print(f"SMAPE: {smape_val:.3f}")
print(f"MAE: {mae_val:.3f}")


# Visualisations des résultats


# On formate les graphiques 
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk", font_scale=1)
sns.set_palette("muted")

# Représentation temporel d’un county
cfips_example = 1001 # 
train_cfips = train[train.cfips == cfips_example].copy()  # Données d’entraînement pour ce comté
pred_cfips = submission[submission.row_id.str.startswith(str(cfips_example))].copy()  # Prédictions correspondantes
pred_cfips["first_day_of_month"] = pd.to_datetime(pred_cfips["row_id"].str[-10:])  # On extrait la date depuis row_id

# On combine les données historiques et prédites pour que ce soit continu
all_cfips = pd.concat([
    train_cfips[["first_day_of_month", "microbusiness_density"]].assign(Source="Train"),
    pred_cfips[["first_day_of_month", "microbusiness_density"]].assign(Source="Forecast")
])

plt.figure(figsize=(11, 6))
sns.lineplot(  
    data=all_cfips,
    x="first_day_of_month",
    y="microbusiness_density",
    hue="Source",
    style="Source",
    markers=True,
    dashes=False,
    linewidth=2.5,
    palette={"Train": "#711fb4", "Forecast": "#788dc1"},
)

forecast_start = pred_cfips["first_day_of_month"].min()
plt.axvspan(forecast_start, all_cfips["first_day_of_month"].max(),
            color="#624c7c", alpha=0.08, label="Période de prévision")

plt.title(f"📈 Prévision de la densité des microentreprises — county {cfips_example}",
          fontsize=17, fontweight="bold", pad=15)
plt.xlabel("Date", fontsize=13)
plt.ylabel("Densité des microentreprises", fontsize=13)
plt.legend(frameon=True, loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Représentaiton de l'importance des variables

imp = pd.DataFrame({  
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)  

plt.figure(figsize=(8, 6))
sns.barplot(  
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


#  Nuage de points prédit vs réel pour la validation
val_results = pd.DataFrame({  
    "Réelles": val_df["microbusiness_density"],
    "Prédites": val_pred_density
})

plt.figure(figsize=(7, 7))
sns.scatterplot(  
    data=val_results,
    x="Réelles",
    y="Prédites",
    alpha=0.6,
    s=50,
    color="#654B72",
    edgecolor="white",
    linewidth=0.5
)


plt.title("Validation — Valeurs prédites vs réelles",
    fontsize=17, fontweight="bold", pad=15)
plt.xlabel("Valeurs réelles", fontsize=13)
plt.ylabel("Valeurs prédites", fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()