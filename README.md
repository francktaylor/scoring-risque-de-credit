# Scoring du risque de défaut de crédit – Home Credit

Projet de bout en bout sur les données [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) : estimer, au moment d'une demande de crédit, la probabilité qu'un client rencontre des difficultés de remboursement, et en déduire une règle d'octroi.

## Résultats

| Étape | AUC | Gini |
|---|---|---|
| Scores externes seuls | 0,717 | 0,43 |
| Régression logistique, table principale | 0,747 | 0,49 |
| Grille de score (20 variables, barème en points) | 0,744 | 0,49 |
| Régression logistique, 7 tables | 0,772 | 0,54 |
| **LightGBM, 7 tables (échantillon de test)** | **0,792** | **0,58** |

- Règle d'octroi fondée sur les coûts : refus au-delà de 17 % de probabilité de défaut. Elle accepte 88 % des demandes, fait passer le taux de défaut des clients acceptés de 8,1 % à 5,2 % et améliore le résultat de 10 % par rapport à une acceptation systématique.
- Les probabilités sont bien calibrées (score de Brier de 0,065).
- Chaque décision est expliquée par ses principaux facteurs (valeurs SHAP).
- Un audit d'équité compare les décisions selon le sexe et l'âge.

**Application interactive :** *lien à ajouter après le déploiement sur Streamlit Community Cloud*

## Démarche

| Notebook | Contenu |
|---|---|
| `01_exploration` | Découpage apprentissage / test fixé dès le départ ; cible déséquilibrée ; mécanismes des valeurs manquantes ; anomalies ; Information Value et AUC univariée ; règles de nettoyage |
| `02_nettoyage_reference` | Nettoyage ; régression logistique de référence ; effet de la pondération des classes sur la calibration ; grille de score WoE avec signes cohérents et conversion en points |
| `03_bureau_sql` | Agrégation en SQL (DuckDB) de 29 millions de lignes du bureau de crédit : jointures, `FILTER`, fonctions de fenêtre |
| `04_historique_home_credit_sql` | Agrégation en SQL des demandes antérieures, remboursements, prêts et cartes ; gain de performance mesuré table par table |
| `05_modele_final` | LightGBM ; évaluation sur le test ; seuil d'octroi selon les coûts ; SHAP ; audit d'équité ; enregistrement du modèle |

## Structure

```
projet-home-credit/
├── data/
│   ├── brutes/        # fichiers CSV de Kaggle (non versionnés)
│   └── traitees/      # découpage et tables intermédiaires produites par les notebooks
├── notebooks/         # 01 à 05, à exécuter dans l'ordre
├── src/
│   ├── preparation.py # règles de nettoyage, écrêtage, encodage WoE, logistique à signes contraints
│   └── modelisation.py# pipeline de référence et validation croisée
├── modeles/           # modèle LightGBM enregistré
├── app/
│   ├── app.py                 # application Streamlit
│   ├── preparer_resultats.py  # génère les résultats agrégés lus par l'application
│   └── resultats/             # tables agrégées (aucune donnée brute)
└── requirements.txt
```

## Exécution

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Placer les fichiers CSV de Kaggle dans `data/brutes/`, puis exécuter les notebooks dans l'ordre. Chacun produit les fichiers nécessaires au suivant.

Pour l'application, depuis la racine du projet :

```bash
python app/preparer_resultats.py
streamlit run app/app.py
```

L'application ne lit que les résultats agrégés de `app/resultats/` : elle fonctionne sans les données de Kaggle, qui ne peuvent pas être republiées.

## Outils

Python (pandas, scikit-learn, LightGBM, SHAP), SQL (DuckDB), matplotlib, seaborn, Streamlit et Altair.
