"""Prépare les fichiers de résultats lus par l'application Streamlit.

À lancer une fois, depuis la racine du projet, après avoir exécuté les notebooks 01 à 05 :

    python app/preparer_resultats.py

Le script recharge les tables préparées et le modèle enregistré, calcule les prédictions sur
l'échantillon test et les valeurs SHAP, puis écrit dans app/resultats/ des tables agrégées.
Aucune donnée brute de Kaggle n'est copiée : seulement des agrégats et quelques fiches résumées.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

RACINE = Path(__file__).resolve().parents[1]
sys.path.append(str(RACINE))
sys.path.append(str(RACINE / "app"))
from libelles import libelle, source  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
TRAITEES = RACINE / "data" / "traitees"
SORTIE = RACINE / "app" / "resultats"
SORTIE.mkdir(parents=True, exist_ok=True)
GRAINE = 42


def charger_base() -> pd.DataFrame:
    application = pd.read_parquet(TRAITEES / "application_propre.parquet")
    bureau = pd.read_parquet(TRAITEES / "variables_bureau.parquet")
    historique = pd.read_parquet(TRAITEES / "variables_historique.parquet")
    return (application.merge(bureau, on="SK_ID_CURR", how="left", validate="one_to_one")
                       .merge(historique, on="SK_ID_CURR", how="left", validate="one_to_one"))


def preparer_lgb(X, qualitatives, categories):
    X = X.copy()
    for c in qualitatives:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def main():
    artefact = joblib.load(RACINE / "modeles" / "modele_lightgbm.joblib")
    modele, numeriques, qualitatives = artefact["modele"], artefact["numeriques"], artefact["qualitatives"]

    base = charger_base()
    for c in numeriques:
        base[c] = base[c].astype(float)
    test = base[base["echantillon"] == "test"].reset_index(drop=True)
    n_app = int((base["echantillon"] == "apprentissage").sum())
    X_test = preparer_lgb(test[numeriques + qualitatives], qualitatives, artefact["categories"])
    y = test["TARGET"].to_numpy()
    proba = modele.predict_proba(X_test)[:, 1]
    print(f"Prédictions sur {len(test):,} clients du test".replace(",", " "))

    # 1. Chiffres clés
    auc = roc_auc_score(y, proba)
    resume = {
        "n_apprentissage": n_app, "n_test": int(len(test)), "taux_defaut": float(y.mean()),
        "auc": float(auc), "gini": float(2 * auc - 1), "auc_pr": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)), "nb_variables": len(numeriques) + len(qualitatives),
        "nb_arbres": int(modele.n_estimators), "seuil": float(artefact["seuil"]),
        "marge": float(artefact["marge"]), "perte": float(artefact["perte"]),
    }
    (SORTIE / "resume.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")

    # 2. Courbe ROC (sous-échantillonnée) et calibration
    fpr, tpr, _ = roc_curve(y, proba)
    pas = max(1, len(fpr) // 300)
    pd.DataFrame({"faux_positifs": fpr[::pas], "vrais_positifs": tpr[::pas]}).to_csv(SORTIE / "roc.csv", index=False)
    observe, predit = calibration_curve(y, proba, n_bins=15, strategy="quantile")
    pd.DataFrame({"predit": predit, "observe": observe}).to_csv(SORTIE / "calibration.csv", index=False)

    # 3. Taux de défaut par décile de risque
    deciles = pd.qcut(proba, 10, labels=False, duplicates="drop") + 1
    (pd.DataFrame({"decile": deciles, "proba": proba, "defaut": y})
       .groupby("decile").agg(clients=("defaut", "size"), proba_min=("proba", "min"), proba_max=("proba", "max"),
                              proba_moyenne=("proba", "mean"), taux_defaut=("defaut", "mean"))
       .reset_index().to_csv(SORTIE / "deciles.csv", index=False))

    # 4. Table agrégée par tranche de score, sexe et âge : base de la règle d'octroi et de l'audit
    montant = (test["AMT_GOODS_PRICE"].fillna(test["AMT_GOODS_PRICE"].median())
               * test["PART_FINANCEE"].fillna(1)).to_numpy()
    tranche = np.minimum((proba / 0.005).astype(int), 199)
    age = pd.cut(test["AGE"], [0, 30, 45, 60, 200], right=False,
                 labels=["moins de 30 ans", "30-45 ans", "45-60 ans", "60 ans et plus"]).astype(str)
    sexe = test["CODE_GENDER"].map({"F": "Femmes", "M": "Hommes"}).fillna("Non renseigné")
    (pd.DataFrame({"tranche": tranche, "sexe": sexe, "age": age, "defaut": y,
                   "montant_bons": montant * (y == 0), "montant_defauts": montant * (y == 1)})
       .groupby(["tranche", "sexe", "age"], observed=True)
       .agg(clients=("defaut", "size"), defauts=("defaut", "sum"),
            montant_bons=("montant_bons", "sum"), montant_defauts=("montant_defauts", "sum"))
       .reset_index().assign(seuil_bas=lambda d: d["tranche"] * 0.005)
       .to_csv(SORTIE / "grille_scores.csv", index=False))

    # 5. SHAP sur un échantillon de 5 000 clients
    echantillon = X_test.sample(min(5000, len(X_test)), random_state=GRAINE)
    proba_ech = proba[echantillon.index.to_numpy()]
    y_ech = y[echantillon.index.to_numpy()]
    valeurs = shap.TreeExplainer(modele)(echantillon)
    if valeurs.values.ndim == 3:
        valeurs = valeurs[:, :, 1]
    contributions = pd.DataFrame(valeurs.values, columns=echantillon.columns)
    base_logit = float(np.ravel(valeurs.base_values)[0])

    importance = contributions.abs().mean().sort_values(ascending=False).head(20)
    pd.DataFrame({"variable": importance.index, "libelle": [libelle(v) for v in importance.index],
                  "source": [source(v) for v in importance.index], "importance": importance.values}
                 ).to_csv(SORTIE / "importance.csv", index=False)

    # 6. Sens de l'effet des principales variables numériques : SHAP moyen par décile de la variable
    lignes = []
    principales = [v for v in importance.index if v in numeriques][:8]
    for v in principales:
        valeurs_v = echantillon[v].reset_index(drop=True)
        connues = valeurs_v.notna()
        groupes = pd.qcut(valeurs_v[connues], 10, labels=False, duplicates="drop")
        d = pd.DataFrame({"valeur": valeurs_v[connues], "shap": contributions.loc[connues, v], "groupe": groupes})
        for _, g in d.groupby("groupe"):
            lignes.append({"variable": v, "libelle": libelle(v), "valeur": g["valeur"].median(),
                           "shap": g["shap"].mean(), "clients": len(g)})
    pd.DataFrame(lignes).to_csv(SORTIE / "effets.csv", index=False)

    # 7. Fiches de quelques clients, du plus sûr au plus risqué
    cibles = [0.02, 0.05, 0.09, 0.14, 0.19, 0.25, 0.35, 0.50]
    choisis = []
    for c in cibles:
        ordre = np.argsort(np.abs(proba_ech - c))
        choisis.append(next(i for i in ordre if i not in choisis))
    fiches = []
    for numero, i in enumerate(choisis, start=1):
        contrib = contributions.iloc[i]
        principaux = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        facteurs = []
        for v, s in principaux.items():
            valeur = echantillon.iloc[i][v]
            valeur = None if pd.isna(valeur) else (str(valeur) if v in qualitatives else float(valeur))
            facteurs.append({"variable": v, "libelle": libelle(v), "valeur": valeur, "contribution": float(s)})
        fiches.append({"client": f"Client {numero}", "proba": float(proba_ech[i]), "defaut_reel": int(y_ech[i]),
                       "base_logit": base_logit, "autres": float(contrib.sum() - principaux.sum()),
                       "facteurs": facteurs})
    (SORTIE / "clients.json").write_text(json.dumps(fiches, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Résultats écrits dans {SORTIE}")
    print(f"AUC test = {auc:.4f}   Gini = {2 * auc - 1:.4f}")


if __name__ == "__main__":
    main()
