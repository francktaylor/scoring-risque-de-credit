"""Garde les applications Streamlit éveillées.

Ouvre chaque application dans un navigateur invisible (Chromium via Playwright)
et clique sur le bouton de réveil si l'application s'est endormie.
Lancé automatiquement par GitHub Actions (.github/workflows/keep-alive.yml).
"""
import re
import sys

from playwright.sync_api import sync_playwright

APPLICATIONS = [
    "https://scoring-credit-francktaylor.streamlit.app",
    "https://classification-genres-musicaux-francktaylor.streamlit.app",
]

BOUTON_REVEIL = re.compile(r"get this app back up", re.IGNORECASE)


def visiter(page, url):
    page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(15_000)
    bouton = page.get_by_role("button", name=BOUTON_REVEIL)
    if bouton.count() > 0:
        print(f"[endormie] {url} -> clic sur le bouton de réveil")
        bouton.first.click()
        page.wait_for_timeout(90_000)  # laisse le temps au redémarrage
        return "réveillée"
    print(f"[éveillée] {url}")
    return "déjà éveillée"


def main():
    erreurs = 0
    with sync_playwright() as p:
        navigateur = p.chromium.launch()
        for url in APPLICATIONS:
            page = navigateur.new_page()
            try:
                visiter(page, url)
            except Exception as exc:  # on continue avec l'application suivante
                erreurs += 1
                print(f"[erreur] {url} : {exc}")
            finally:
                page.close()
        navigateur.close()
    sys.exit(1 if erreurs else 0)


if __name__ == "__main__":
    main()
