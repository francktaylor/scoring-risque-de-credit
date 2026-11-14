"""Préparation des données Home Credit.

Deux types de traitements :
- `nettoyer` applique des règles fixes, qui ne dépendent pas des données :
  elles peuvent être appliquées à toute la base sans fuite d'information ;
- les transformateurs `Ecreteur` et `EncodeurWoE` apprennent des paramètres
  (centiles, classes, WoE) : ils s'utilisent dans un pipeline scikit-learn,
  ajustés sur l'échantillon d'apprentissage uniquement.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression

CODE_SANS_EMPLOI = 365243
VARIABLES_EQUITE = ["CODE_GENDER", "NAME_FAMILY_STATUS"]

# Paires quasi identiques ou variables sans information : on ne garde qu'une version
A_SUPPRIMER = [
    "OBS_60_CNT_SOCIAL_CIRCLE",   # identique à OBS_30 (r = 0,999)
    "REGION_RATING_CLIENT",       # quasi identique à la version _W_CITY (r = 0,95)
    "AMT_CREDIT",                 # identique à AMT_GOODS_PRICE (r = 0,99) ; l'écart est porté par PART_FINANCEE
    "FLAG_MOBIL",                 # constante (une seule demande à 0)
    "FONDKAPREMONT_MODE",         # 68 % de manquants, sens inconnu
]


def nettoyer(brut: pd.DataFrame) -> pd.DataFrame:
    """Règles de nettoyage fixes issues du notebook d'exploration."""
    df = brut.copy()

    # 1. Codes d'absence -> valeurs manquantes
    df["SANS_EMPLOI"] = (df["DAYS_EMPLOYED"] == CODE_SANS_EMPLOI).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(CODE_SANS_EMPLOI, np.nan)
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
    df["NAME_FAMILY_STATUS"] = df["NAME_FAMILY_STATUS"].replace("Unknown", np.nan)
    df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace("XNA", "Sans employeur")

    # 2. Absences structurelles : la valeur n'existe pas
    retraite = df["NAME_INCOME_TYPE"].eq("Pensioner")
    df.loc[df["OCCUPATION_TYPE"].isna() & retraite, "OCCUPATION_TYPE"] = "Retraité"
    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Non renseignée")
    # OWN_CAR_AGE reste manquant pour les clients sans voiture : FLAG_OWN_CAR porte l'information

    # 3. Durées en années positives
    durees = {"DAYS_BIRTH": "AGE", "DAYS_EMPLOYED": "ANCIENNETE_EMPLOI",
              "DAYS_REGISTRATION": "ANCIENNETE_ENREGISTREMENT", "DAYS_ID_PUBLISH": "ANCIENNETE_PIECE_IDENTITE",
              "DAYS_LAST_PHONE_CHANGE": "ANCIENNETE_TELEPHONE"}
    for origine, nouveau in durees.items():
        df[nouveau] = -df[origine] / 365.25
    df = df.drop(columns=list(durees))

    # 4. Binaires textuelles -> 0/1
    df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].map({"Y": 1, "N": 0})
    df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0})
    df["CREDIT_RENOUVELABLE"] = df.pop("NAME_CONTRACT_TYPE").eq("Revolving loans").astype(int)
    df["EMERGENCYSTATE_MODE"] = df["EMERGENCYSTATE_MODE"].map({"Yes": 1, "No": 0})

    # 5. Variables d'immeuble : versions _AVG + surface totale + indicatrice de manque
    immeuble = [c for c in df.columns if c.endswith(("_AVG", "_MODE", "_MEDI"))
                and df[c].dtype.kind == "f" and c not in ("TOTALAREA_MODE", "EMERGENCYSTATE_MODE")]
    moyennes = [c for c in immeuble if c.endswith("_AVG")]
    df["NB_INFOS_IMMEUBLE_MANQUANTES"] = df[moyennes].isna().sum(axis=1)
    df = df.drop(columns=[c for c in immeuble if not c.endswith("_AVG")])

    # 6. Documents : seul le document 3 est fréquent ; les autres sont résumés par un compteur
    documents = [c for c in df.columns if c.startswith("FLAG_DOCUMENT")]
    df["NB_AUTRES_DOCUMENTS"] = df[[c for c in documents if c != "FLAG_DOCUMENT_3"]].sum(axis=1)
    df = df.drop(columns=[c for c in documents if c != "FLAG_DOCUMENT_3"])

    # 7. Variables dérivées
    externes = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["MOYENNE_EXT"] = df[externes].mean(axis=1)
    df["MIN_EXT"] = df[externes].min(axis=1)
    df["NB_EXT_MANQUANTS"] = df[externes].isna().sum(axis=1)
    df["PART_FINANCEE"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["DUREE_ANNUITES"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["TAUX_EFFORT"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_REVENU"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["REVENU_PAR_PERSONNE"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANCIENNETE_SUR_AGE"] = df["ANCIENNETE_EMPLOI"] / df["AGE"]

    return df.drop(columns=A_SUPPRIMER)


class Ecreteur(BaseEstimator, TransformerMixin):
    """Ramène chaque variable dans l'intervalle [q_bas, q_haut] appris sur l'apprentissage."""

    def __init__(self, q_bas=0.001, q_haut=0.999):
        self.q_bas = q_bas
        self.q_haut = q_haut

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.bornes_ = X.quantile([self.q_bas, self.q_haut])
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(float)
        return X.clip(self.bornes_.iloc[0], self.bornes_.iloc[1], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.bornes_.columns)


class EncodeurWoE(BaseEstimator, TransformerMixin):
    """Remplace chaque variable par le Weight of Evidence de sa classe.

    Variables numériques : découpage en `nb_classes` quantiles appris sur l'apprentissage.
    Variables qualitatives : une classe par modalité, les modalités rares (< `freq_min`) regroupées.
    Les valeurs manquantes forment une classe à part. Convention : WoE = ln(% bons / % mauvais),
    donc une WoE positive signale une classe moins risquée que la moyenne.

    Sélection optionnelle (apprise elle aussi sur l'apprentissage) : on écarte les variables
    d'IV inférieure à `iv_min`, puis, parmi les paires de variables WoE corrélées à plus de
    `corr_max`, on garde celle qui a la plus forte IV.
    """

    def __init__(self, nb_classes=10, freq_min=0.01, lissage=0.5, iv_min=0.0, corr_max=1.0):
        self.nb_classes = nb_classes
        self.freq_min = freq_min
        self.lissage = lissage
        self.iv_min = iv_min
        self.corr_max = corr_max

    def _classes(self, nom, serie):
        if nom in self.bornes_:
            c = pd.cut(serie, self.bornes_[nom], include_lowest=True).astype(str)
        else:
            c = serie.astype(str).where(serie.astype(str).isin(self.modalites_[nom]), "autres")
        return c.where(serie.notna(), "manquant")

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y), index=X.index)
        self.bornes_, self.modalites_, self.woe_, self.iv_ = {}, {}, {}, {}
        for nom in X.columns:
            serie = X[nom]
            if serie.dtype.kind in "biuf" and serie.nunique() > self.nb_classes:
                q = np.unique(serie.quantile(np.linspace(0, 1, self.nb_classes + 1)).values.astype(float))
                q[0], q[-1] = -np.inf, np.inf
                self.bornes_[nom] = q
            else:
                frequences = serie.astype(str).value_counts(normalize=True)
                self.modalites_[nom] = set(frequences[frequences >= self.freq_min].index)
            t = pd.crosstab(self._classes(nom, serie), y) + self.lissage
            bons, mauvais = t[0] / t[0].sum(), t[1] / t[1].sum()
            woe = np.log(bons / mauvais)
            self.woe_[nom] = woe.to_dict()
            self.iv_[nom] = float(((bons - mauvais) * woe).sum())
        self.colonnes_ = list(X.columns)

        # Sélection : IV minimale, puis élimination des redondances
        candidates = sorted((n for n in self.colonnes_ if self.iv_[n] >= self.iv_min),
                            key=lambda n: -self.iv_[n])
        if self.corr_max < 1.0 and candidates:
            woe_app = self._encoder(X, candidates)
            correlations = woe_app.corr().abs()
            retenues = []
            for n in candidates:
                if all(correlations.loc[n, r] <= self.corr_max for r in retenues):
                    retenues.append(n)
            candidates = retenues
        self.retenues_ = candidates
        return self

    def _encoder(self, X, colonnes):
        sortie = pd.DataFrame(index=X.index)
        for nom in colonnes:
            sortie[nom] = self._classes(nom, X[nom]).map(self.woe_[nom]).fillna(0.0).astype(float)
        return sortie

    def transform(self, X):
        return self._encoder(pd.DataFrame(X), self.retenues_)

    def table_woe(self, nom):
        """Classes, WoE de chaque classe pour une variable (pour l'affichage)."""
        return pd.Series(self.woe_[nom], name="WoE").rename_axis("classe")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.retenues_)


class LogistiqueSignesCoherents(ClassifierMixin, BaseEstimator):
    """Régression logistique sur des WoE, avec signes imposés.

    Avec la convention WoE = ln(% bons / % mauvais), une WoE élevée signale une classe sûre :
    dans un modèle de la probabilité de défaut, tous les coefficients doivent donc être négatifs.
    Un coefficient positif (dû à la colinéarité entre variables) rendrait la grille incohérente :
    une classe plus sûre rapporterait moins de points. On retire alors la variable au coefficient
    le plus positif et on réestime, jusqu'à ce que tous les signes soient cohérents.
    """

    def __init__(self, C=1.0, max_iter=2000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        X = pd.DataFrame(X)
        colonnes = list(X.columns)
        self.retirees_ = []
        while True:
            modele = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(X[colonnes], y)
            coef = pd.Series(modele.coef_[0], index=colonnes)
            if (coef <= 0).all() or len(colonnes) == 1:
                break
            pire = coef.idxmax()
            self.retirees_.append(pire)
            colonnes.remove(pire)
        self.modele_, self.colonnes_ = modele, colonnes
        self.coef_, self.intercept_ = modele.coef_[0], modele.intercept_[0]
        self.classes_ = modele.classes_
        return self

    def predict_proba(self, X):
        return self.modele_.predict_proba(pd.DataFrame(X)[self.colonnes_])

    def predict(self, X):
        return self.modele_.predict(pd.DataFrame(X)[self.colonnes_])
