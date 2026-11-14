"""Outils de modélisation communs aux notebooks."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preparation import Ecreteur

GRAINE = 42
VALIDATION = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)
METRIQUES = {"auc": "roc_auc", "auc_pr": "average_precision", "brier": "neg_brier_score"}


def pipeline_logistique(numeriques, qualitatives, C=0.01):
    """Régression logistique de référence (notebook 02) : écrêtage, imputation, standardisation, one-hot."""
    pre = ColumnTransformer([
        ("num", Pipeline([("ecretage", Ecreteur()),
                          ("imputation", SimpleImputer(strategy="median", add_indicator=True)),
                          ("standardisation", StandardScaler())]), numeriques),
        ("qual", Pipeline([("imputation", SimpleImputer(strategy="constant", fill_value="manquant")),
                           ("encodage", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005))]),
         qualitatives),
    ])
    return Pipeline([("preparation", pre), ("modele", LogisticRegression(C=C, max_iter=3000))])


def evaluer(modele, X, y, nom, cv=VALIDATION, n_jobs=-1):
    """Validation croisée stratifiée : AUC, Gini, AUC précision-rappel et score de Brier."""
    r = cross_validate(modele, X, y, cv=cv, scoring=METRIQUES, n_jobs=n_jobs)
    return {"modèle": nom, "AUC": r["test_auc"].mean(), "écart-type AUC": r["test_auc"].std(),
            "Gini": 2 * r["test_auc"].mean() - 1, "AUC PR": r["test_auc_pr"].mean(),
            "Brier": -r["test_brier"].mean()}
