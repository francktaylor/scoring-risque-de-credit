"""Application Streamlit : score d'octroi de crédit (données Home Credit).

Lancement en local, depuis la racine du projet :  streamlit run app/app.py
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from libelles import libelle

RESULTATS = Path(__file__).parent / "resultats"
BLEU, ROUGE, VERT, GRIS = "#2E5E8C", "#C44E52", "#4C9A6A", "#8C8C8C"

st.set_page_config(page_title="Score d'octroi de crédit", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- données
@st.cache_data
def charger():
    lire = lambda nom: pd.read_csv(RESULTATS / nom)  # noqa: E731
    clients = json.loads((RESULTATS / "clients.json").read_text(encoding="utf-8"))
    for c in clients:
        for f in c["facteurs"]:
            f["libelle"] = libelle(f["variable"])
    importance, effets = lire("importance.csv"), lire("effets.csv")
    importance["libelle"] = importance["variable"].map(libelle)
    effets["libelle"] = effets["variable"].map(libelle)
    return {
        "resume": json.loads((RESULTATS / "resume.json").read_text(encoding="utf-8")),
        "clients": clients,
        "grille": lire("grille_scores.csv"), "roc": lire("roc.csv"), "calibration": lire("calibration.csv"),
        "deciles": lire("deciles.csv"), "importance": importance, "effets": effets,
    }


D = charger()
R = D["resume"]

# Résultats des notebooks 02, 04 et 05 (validation croisée à 5 plis sur l'apprentissage)
ETAPES = pd.DataFrame([
    ("Scores externes seuls", "Régression logistique", 0.7167),
    ("Table principale", "Régression logistique", 0.7471),
    ("+ bureau de crédit", "Régression logistique", 0.7525),
    ("+ demandes antérieures", "Régression logistique", 0.7586),
    ("+ remboursements", "Régression logistique", 0.7643),
    ("+ prêts conso / perso", "Régression logistique", 0.7687),
    ("+ cartes de crédit", "Régression logistique", 0.7716),
    ("LightGBM, 7 tables", "LightGBM", 0.7882),
], columns=["étape", "modèle", "AUC"])


def pct(x, d=1):
    return f"{x * 100:.{d}f} %".replace(".", ",")


def nombre(x):
    return f"{x:,.0f}".replace(",", " ")


def decision(grille, seuil, marge, perte):
    """Résultat d'une règle « refus si probabilité de défaut ≥ seuil », à partir de la table agrégée."""
    acc = grille[grille["seuil_bas"] < seuil - 1e-9]
    clients, defauts = grille["clients"].sum(), grille["defauts"].sum()
    resultat = marge * acc["montant_bons"].sum() - perte * acc["montant_defauts"].sum()
    return {
        "seuil": seuil,
        "taux_acceptation": acc["clients"].sum() / clients,
        "taux_defaut_acceptes": acc["defauts"].sum() / max(acc["clients"].sum(), 1),
        "defauts_evites": 1 - acc["defauts"].sum() / defauts,
        "bons_refuses": 1 - (acc["clients"].sum() - acc["defauts"].sum()) / (clients - defauts),
        "resultat": resultat,
    }


def courbe_decision(grille, marge, perte):
    seuils = np.round(np.arange(0.02, 0.605, 0.005), 3)
    return pd.DataFrame([decision(grille, s, marge, perte) for s in seuils])


# ---------------------------------------------------------------- navigation
PAGES = ["Le projet", "Construction du modèle", "Performance", "Règle d'octroi",
         "Explication des décisions", "Équité", "Limites et suites"]
with st.sidebar:
    st.markdown("### Score d'octroi de crédit")
    st.caption("Probabilité de défaut et décision d'octroi, à partir des données Home Credit")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption(f"Modèle : LightGBM ({R['nb_variables']} variables)\n\n"
               f"Test : AUC {R['auc']:.3f} · Gini {R['gini']:.2f}".replace(".", ","))
    st.divider()
    st.link_button("💻 Voir le code sur GitHub",
                   "https://github.com/francktaylor/scoring-risque-de-credit",
                   width="stretch")
    st.caption("Réalisé par **Franck Taylor MBESSO MAGOUA**, élève ingénieur à l'ENSAI  \n"
               "[LinkedIn](https://www.linkedin.com/in/franck-taylor-mbesso) · "
               "[GitHub](https://github.com/francktaylor)")

if "seuil" not in st.session_state:
    st.session_state.update(seuil=R["seuil"], marge=R["marge"], perte=R["perte"])


# ---------------------------------------------------------------- pages
if page == "Le projet":
    st.title("Prévoir le risque de défaut au moment de la demande de crédit")
    st.markdown(
        "Une société de crédit à la consommation reçoit une demande de prêt. Faut-il l'accepter ? "
        "Ce projet estime la **probabilité que le client rencontre des difficultés de remboursement**, "
        "à partir de sa demande et de tout son historique de crédit, puis en déduit une **règle d'octroi** "
        "fondée sur les gains et les pertes attendus.")
    c = st.columns(4)
    c[0].metric("Demandes de crédit", nombre(R["n_apprentissage"] + R["n_test"]))
    c[1].metric("Taux de défaut moyen", pct(R["taux_defaut"]))
    c[2].metric("AUC sur le test", f"{R['auc']:.3f}".replace(".", ","))
    c[3].metric("Gini sur le test", f"{R['gini']:.2f}".replace(".", ","))
    reference = decision(D["grille"], 1.01, R["marge"], R["perte"])
    regle = decision(D["grille"], R["seuil"], R["marge"], R["perte"])
    c = st.columns(4)
    c[0].metric("Seuil d'octroi", f"{pct(R['seuil'], 0)} de risque")
    c[1].metric("Demandes acceptées", pct(regle["taux_acceptation"], 0))
    c[2].metric("Défaut des acceptés", pct(regle["taux_defaut_acceptes"]),
                f"{(regle['taux_defaut_acceptes'] - reference['taux_defaut_acceptes']) * 100:+.1f} points".replace(".", ","),
                delta_color="inverse")
    c[3].metric("Résultat vs « tout accepter »",
                f"{(regle['resultat'] - reference['resultat']) / abs(reference['resultat']) * 100:+.0f} %")

    st.subheader("La démarche")
    st.table(pd.DataFrame({
        "Étape": ["Données", "Découpage", "Nettoyage", "Variables", "Modèles", "Décision", "Transparence"],
        "Contenu": [
            "7 tables relationnelles, 58 millions de lignes : demande, bureau de crédit, demandes antérieures, "
            "remboursements, prêts à la consommation, cartes",
            "Test de 20 % mis de côté dès le départ et utilisé une seule fois, à la fin",
            "Anomalies, mécanismes des valeurs manquantes, écrêtage appris sur l'apprentissage seul",
            "Agrégation en SQL (DuckDB) de l'historique de chaque client en 150 variables",
            "Régression logistique de référence, grille de score (WoE), puis LightGBM",
            "Seuil d'octroi choisi en maximisant le résultat financier attendu",
            "Motifs de chaque décision (SHAP) et audit d'équité selon le sexe et l'âge",
        ]}).set_index("Étape"))

elif page == "Construction du modèle":
    st.title("Ce que chaque source d'information apporte")
    st.markdown(
        "Les sources sont ajoutées une à une à la régression logistique, avec les mêmes plis de validation "
        "croisée, pour mesurer le gain propre à chacune. LightGBM est ensuite entraîné sur l'ensemble.")
    donnees = ETAPES.rename(columns={"étape": "etape", "modèle": "modele", "AUC": "auc"})
    ordre = donnees["etape"].tolist()
    base = alt.Chart(donnees).encode(
        y=alt.Y("etape:N", sort=ordre, title=None, axis=alt.Axis(labelLimit=300, labelFontSize=13)),
        x=alt.X("auc:Q", scale=alt.Scale(domain=[0.70, 0.80], clamp=True), title="AUC (validation croisée)"),
        tooltip=[alt.Tooltip("etape:N", title="étape"), alt.Tooltip("modele:N", title="modèle"),
                 alt.Tooltip("auc:Q", format=".4f", title="AUC")])
    barres = base.mark_bar(clip=True).encode(
        color=alt.Color("modele:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Régression logistique", "LightGBM"], range=[GRIS, BLEU])))
    texte = base.mark_text(align="left", dx=4, color="black").encode(text=alt.Text("auc:Q", format=".3f"))
    st.altair_chart((barres + texte).properties(height=320), width="stretch")
    c = st.columns(3)
    c[0].markdown("**Les scores externes dominent.** Seuls, ils atteignent déjà une AUC de 0,717 : ils résument "
                  "probablement l'historique du client auprès des bureaux de crédit.")
    c[1].markdown("**L'historique interne apporte plus que le bureau externe.** Demandes, remboursements, "
                  "prêts et cartes chez Home Credit font gagner 1,9 point d'AUC, contre 0,5 pour le bureau.")
    c[2].markdown("**LightGBM ajoute 1,7 point.** Il capte ce que le modèle linéaire ne peut pas : "
                  "effets non monotones, seuils et interactions.")
    st.info("Une **grille de score** classique (20 variables encodées en WoE, coefficients de signe cohérent, "
            "barème en points) atteint une AUC de 0,744 : presque autant que la logistique complète, "
            "avec un modèle lisible ligne par ligne.")

elif page == "Performance":
    st.title("Performance sur l'échantillon test")
    st.markdown(f"Évaluation unique sur **{nombre(R['n_test'])} demandes** jamais utilisées pendant la mise au point.")
    c = st.columns(4)
    c[0].metric("AUC", f"{R['auc']:.3f}".replace(".", ","))
    c[1].metric("Gini", f"{R['gini']:.3f}".replace(".", ","))
    c[2].metric("AUC précision-rappel", f"{R['auc_pr']:.3f}".replace(".", ","),
                f"hasard : {R['taux_defaut']:.3f}".replace(".", ","), delta_color="off")
    c[3].metric("Score de Brier", f"{R['brier']:.4f}".replace(".", ","))
    g, d = st.columns(2)
    with g:
        st.subheader("Courbe ROC")
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=GRIS).encode(x="x", y="y")
        roc = alt.Chart(D["roc"]).mark_line(color=BLEU, strokeWidth=2.5).encode(
            x=alt.X("faux_positifs:Q", title="Bons clients refusés (taux de faux positifs)"),
            y=alt.Y("vrais_positifs:Q", title="Défauts détectés (taux de vrais positifs)"))
        st.altair_chart((diag + roc).properties(height=340), width="stretch")
    with d:
        st.subheader("Calibration")
        cal = D["calibration"]
        m = float(max(cal["predit"].max(), cal["observe"].max())) * 1.05
        diag = alt.Chart(pd.DataFrame({"x": [0, m], "y": [0, m]})).mark_line(strokeDash=[4, 4], color=GRIS).encode(x="x", y="y")
        pts = alt.Chart(cal).mark_line(point=True, color=BLEU).encode(
            x=alt.X("predit:Q", title="Probabilité de défaut prédite", axis=alt.Axis(format="%")),
            y=alt.Y("observe:Q", title="Taux de défaut observé", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("predit:Q", format=".1%"), alt.Tooltip("observe:Q", format=".1%")])
        st.altair_chart((diag + pts).properties(height=340), width="stretch")
    st.caption("Les points suivent la diagonale : quand le modèle annonce 15 % de risque, environ 15 % des clients "
               "font défaut. Ces probabilités peuvent donc servir directement à calculer des pertes attendues.")
    st.subheader("Taux de défaut par décile de risque")
    dec = D["deciles"].assign(libelle=lambda x: "D" + x["decile"].astype(str))
    barres = alt.Chart(dec).mark_bar(color=BLEU).encode(
        x=alt.X("libelle:N", sort=None, title="Décile de probabilité prédite (D1 : les moins risqués)"),
        y=alt.Y("taux_defaut:Q", title="Taux de défaut observé", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("clients:Q", format=","), alt.Tooltip("taux_defaut:Q", format=".1%"),
                 alt.Tooltip("proba_moyenne:Q", format=".1%", title="probabilité moyenne")])
    st.altair_chart(barres.properties(height=280), width="stretch")
    premier, dernier = dec["taux_defaut"].iloc[0], dec["taux_defaut"].iloc[-1]
    st.caption(f"Le taux de défaut passe de {pct(premier)} dans le premier décile à {pct(dernier)} dans le dernier, "
               f"soit un rapport de 1 à {dernier / max(premier, 1e-9):.0f}.")

elif page == "Règle d'octroi":
    st.title("Choisir le seuil d'octroi selon les coûts")
    st.markdown(
        "Un crédit accepté rapporte une **marge** s'il est remboursé et fait perdre une partie du montant "
        "(**perte en cas de défaut**) sinon. Le seuil optimal refuse les demandes dont la perte attendue "
        "dépasse le gain attendu. Modifiez les hypothèses : tout est recalculé sur l'échantillon test.")
    c = st.columns(3)
    marge = c[0].slider("Marge sur un crédit remboursé (%)", 2, 30, round(st.session_state.marge * 100), 1) / 100
    perte = c[1].slider("Perte en cas de défaut, LGD (%)", 10, 90, round(st.session_state.perte * 100), 5) / 100
    courbe = courbe_decision(D["grille"], marge, perte)
    optimal = float(courbe.loc[courbe["resultat"].idxmax(), "seuil"])
    hypotheses_projet = abs(marge - R["marge"]) < 1e-9 and abs(perte - R["perte"]) < 1e-9
    defaut = R["seuil"] if hypotheses_projet else optimal
    seuil = c[2].slider("Seuil de refus : probabilité de défaut (%)", 2.0, 60.0, round(defaut * 100, 1), 0.5,
                        format="%.1f") / 100
    st.session_state.update(seuil=seuil, marge=marge, perte=perte)
    if hypotheses_projet:
        st.caption(f"Seuil retenu dans le projet : **{pct(R['seuil'], 0)}**, choisi sur les prédictions hors-pli de "
                   f"l'apprentissage, puis appliqué tel quel au test. Sur le test, le meilleur seuil serait {pct(optimal, 1)} : "
                   "la courbe est presque plate dans cette zone.")
    else:
        st.caption(f"Seuil qui maximise le résultat sur le test pour ces hypothèses : **{pct(optimal, 1)}**. "
                   f"Ordre de grandeur théorique : marge / (marge + perte) = {pct(marge / (marge + perte), 1)}.")

    r = decision(D["grille"], seuil, marge, perte)
    tout = decision(D["grille"], 1.01, marge, perte)
    c = st.columns(5)
    c[0].metric("Demandes acceptées", pct(r["taux_acceptation"]))
    c[1].metric("Défaut des acceptés", pct(r["taux_defaut_acceptes"]),
                f"{(r['taux_defaut_acceptes'] - tout['taux_defaut_acceptes']) * 100:+.1f} points".replace(".", ","),
                delta_color="inverse")
    c[2].metric("Défauts évités", pct(r["defauts_evites"], 0))
    c[3].metric("Bons clients refusés", pct(r["bons_refuses"], 0))
    c[4].metric("Résultat vs « tout accepter »",
                f"{(r['resultat'] - tout['resultat']) / abs(tout['resultat']) * 100:+.1f} %".replace(".", ","))

    courbe["resultat_relatif"] = courbe["resultat"] / abs(tout["resultat"])
    base = alt.Chart(courbe).encode(x=alt.X("seuil:Q", title="Seuil de refus", axis=alt.Axis(format="%")))
    ligne = base.mark_line(color=BLEU, strokeWidth=2.5).encode(
        y=alt.Y("resultat_relatif:Q", title="Résultat (1 = tout accepter)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("seuil:Q", format=".1%"), alt.Tooltip("taux_acceptation:Q", format=".1%"),
                 alt.Tooltip("resultat_relatif:Q", format=".3f")])
    regle = alt.Chart(pd.DataFrame({"s": [seuil]})).mark_rule(color=ROUGE, strokeDash=[5, 3]).encode(x="s:Q")
    ref = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color=GRIS, strokeDash=[2, 2]).encode(y="y:Q")
    g, d = st.columns(2)
    g.altair_chart((ligne + regle + ref).properties(height=300, title="Résultat financier selon le seuil"),
                   width="stretch")
    volumes = courbe.melt("seuil", ["taux_acceptation", "taux_defaut_acceptes"], var_name="indicateur")
    volumes["indicateur"] = volumes["indicateur"].map({"taux_acceptation": "Taux d'acceptation",
                                                       "taux_defaut_acceptes": "Taux de défaut des acceptés"})
    vol = alt.Chart(volumes).mark_line(strokeWidth=2.5).encode(
        x=alt.X("seuil:Q", title="Seuil de refus", axis=alt.Axis(format="%")),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("indicateur:N", scale=alt.Scale(range=[VERT, ROUGE]), legend=alt.Legend(orient="bottom", title=None)))
    d.altair_chart((vol + regle).properties(height=300, title="Volume accepté et qualité du portefeuille"),
                   width="stretch")
    st.caption("Le montant de chaque crédit est approché par le prix du bien multiplié par la part financée. "
               "La marge et la perte sont des hypothèses : avec les paramètres réels d'une banque, il suffit de "
               "relancer ce calcul.")

elif page == "Explication des décisions":
    st.title("Pourquoi le modèle accepte ou refuse")
    onglets = st.tabs(["Fiches clients", "Variables les plus influentes", "Sens des effets"])
    with onglets[0]:
        st.markdown("Chaque décision se décompose en contributions de variables (valeurs SHAP, sur l'échelle du logit). "
                    "En rouge, ce qui **augmente** le risque ; en vert, ce qui le **diminue**.")
        noms = [f"{c['client']} : risque de {pct(c['proba'])}" for c in D["clients"]]
        choix = st.selectbox("Client de l'échantillon test", noms, index=len(noms) // 2)
        client = D["clients"][noms.index(choix)]
        seuil = st.session_state.seuil
        refuse = client["proba"] >= seuil
        c = st.columns(3)
        c[0].metric("Probabilité de défaut", pct(client["proba"]))
        c[1].metric(f"Décision (seuil de {pct(seuil)})", "Refus" if refuse else "Acceptation")
        c[2].metric("Ce qui s'est réellement passé", "Défaut" if client["defaut_reel"] else "Remboursé")
        f = pd.DataFrame(client["facteurs"])

        def afficher(v):
            if v is None:
                return "non renseigné"
            if isinstance(v, str):
                return v
            return f"{v:,.2f}".replace(",", " ").replace(".", ",") if abs(v) < 1000 else nombre(v)

        f["etiquette"] = f["libelle"] + " = " + f["valeur"].map(afficher)
        f["effet"] = np.where(f["contribution"] > 0, "augmente le risque", "diminue le risque")
        barres = alt.Chart(f.drop(columns="valeur")).mark_bar().encode(
            x=alt.X("contribution:Q", title="Contribution au risque (logit)"),
            y=alt.Y("etiquette:N", sort=alt.EncodingSortField("contribution", op="sum", order="descending"), title=None,
                    axis=alt.Axis(labelLimit=420)),
            color=alt.Color("effet:N", scale=alt.Scale(domain=["augmente le risque", "diminue le risque"], range=[ROUGE, VERT]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["libelle", alt.Tooltip("contribution:Q", format="+.3f")])
        st.altair_chart(barres.properties(height=330), width="stretch")
        motifs = f[f["contribution"] > 0].sort_values("contribution", ascending=False)["libelle"].head(3).tolist()
        if refuse and motifs:
            st.markdown("**Principaux motifs de refus :** " + " ; ".join(motifs) + ".")
        autres = f"{client['autres']:+.2f}".replace(".", ",")
        st.caption(f"Les {len(f)} variables affichées sont les plus influentes pour ce client ; "
                   f"les {R['nb_variables'] - len(f)} autres contribuent ensemble pour {autres} au logit.")
    with onglets[1]:
        imp = D["importance"]
        barres = alt.Chart(imp).mark_bar().encode(
            x=alt.X("importance:Q", title="|valeur SHAP| moyenne"),
            y=alt.Y("libelle:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420)),
            color=alt.Color("source:N", scale=alt.Scale(domain=["Demande de crédit", "Bureau de crédit", "Historique Home Credit"],
                                                        range=[BLEU, GRIS, VERT]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["variable", "libelle", alt.Tooltip("importance:Q", format=".3f")])
        st.altair_chart(barres.properties(height=520), width="stretch")
        st.caption("Calculé sur 5 000 clients du test.")
    with onglets[2]:
        eff = D["effets"]
        choix = st.selectbox("Variable", eff["libelle"].unique())
        e = eff[eff["libelle"] == choix]
        ligne = alt.Chart(e).mark_line(point=True, color=BLEU, strokeWidth=2.5).encode(
            x=alt.X("valeur:Q", title=choix), y=alt.Y("shap:Q", title="Effet moyen sur le risque (logit)"),
            tooltip=[alt.Tooltip("valeur:Q", format=",.3f"), alt.Tooltip("shap:Q", format="+.3f"), "clients"])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GRIS).encode(y="y:Q")
        st.altair_chart((zero + ligne).properties(height=330), width="stretch")
        st.caption("Chaque point regroupe un dixième des clients, classés selon la valeur de la variable. "
                   "Au-dessus de zéro, la variable augmente le risque.")

elif page == "Équité":
    st.title("Audit d'équité des décisions")
    seuil = st.session_state.seuil
    st.markdown(
        f"Le sexe n'est **pas** une variable du modèle. On vérifie pourtant que les décisions, au seuil de "
        f"**{pct(seuil)}** (modifiable dans la page « Règle d'octroi »), ne pénalisent pas un groupe "
        "de façon disproportionnée.")

    def audit(colonne):
        g = D["grille"].assign(refus=lambda x: x["seuil_bas"] >= seuil - 1e-9)
        lignes = []
        for groupe, d in g.groupby(colonne):
            if d["clients"].sum() < 100:  # groupes trop petits (sexe non renseigné) : non interprétables
                continue
            bons = d["clients"] - d["defauts"]
            lignes.append({"groupe": groupe, "clients": int(d["clients"].sum()),
                           "taux de défaut réel": d["defauts"].sum() / d["clients"].sum(),
                           "taux de refus": d.loc[d["refus"], "clients"].sum() / d["clients"].sum(),
                           "bons payeurs refusés à tort": bons[d["refus"]].sum() / bons.sum()})
        t = pd.DataFrame(lignes).set_index("groupe")
        acceptation = 1 - t["taux de refus"]
        return t, acceptation.min() / acceptation.max()

    for titre, colonne, ordre in [("Selon le sexe", "sexe", None),
                                  ("Selon l'âge", "age", ["moins de 30 ans", "30-45 ans", "45-60 ans", "60 ans et plus"])]:
        st.subheader(titre)
        t, ratio = audit(colonne)
        if ordre:
            t = t.reindex([o for o in ordre if o in t.index])
        g, d = st.columns([3, 1])
        g.dataframe(t.style.format({"clients": nombre, "taux de défaut réel": pct, "taux de refus": pct,
                                    "bons payeurs refusés à tort": pct}), width="stretch")
        d.metric("Règle des quatre cinquièmes", f"{ratio:.2f}".replace(".", ","),
                 "au-dessus de 0,8" if ratio >= 0.8 else "sous le seuil de 0,8",
                 delta_color="normal" if ratio >= 0.8 else "inverse")
    st.caption("Règle des quatre cinquièmes : rapport entre le plus faible et le plus fort taux d'acceptation des groupes. "
               "En dessous de 0,8, l'écart est généralement considéré comme un indice d'effet discriminatoire.")
    st.markdown(
        "**Lecture.** Les écarts de refus reflètent en partie des écarts de risque réels, mais à comportement "
        "de remboursement égal, un bon payeur n'a pas la même probabilité d'être refusé selon son groupe. "
        "Pour le sexe, le modèle le retrouve indirectement à travers des variables liées (profession, secteur, "
        "voiture). L'âge, lui, est utilisé directement : c'est courant en crédit, mais cela doit être justifié et suivi.")

else:
    st.title("Limites et suites")
    st.markdown("""
**Biais de sélection.** Le modèle n'apprend que sur des crédits accordés : on ne sait pas comment auraient
remboursé les clients refusés par le passé. Les banques traitent ce problème par des techniques
d'inférence des refusés (*reject inference*).

**Validation dans le temps.** Les données ne contiennent pas de date de demande : le test est tiré au hasard,
et non postérieur à l'apprentissage. En production, il faudrait valider le modèle sur une période plus récente
et suivre la stabilité de sa population (indice de stabilité, PSI).

**Hypothèses de coût.** La marge et la perte en cas de défaut sont des ordres de grandeur, pas des
chiffres de Home Credit. La page « Règle d'octroi » permet de mesurer leur effet sur le seuil.

**Réglage du modèle.** Les paramètres de LightGBM ont été fixés à des valeurs raisonnables, sans recherche
approfondie : une optimisation ferait gagner au mieux quelques millièmes d'AUC.

**Suites possibles.** Un modèle sans la variable d'âge pour mesurer le coût de l'équité ; une grille de score
complète intégrant l'historique ; une API qui renvoie la probabilité, la décision et les motifs d'un client.
""")
    st.caption("Données : compétition Kaggle « Home Credit Default Risk ». Seuls des résultats agrégés sont "
               "publiés dans cette application.")
