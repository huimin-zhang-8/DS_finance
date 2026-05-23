"""
TP6 — Scraping de news financières via NewsAPI.

Fonctions exportées :
    load_existing_news, get_news_by_date, scrape_all_companies_news
"""

import os
import json
import time
import requests

from datetime import datetime, timedelta

from configs.settings import NEWS_API_KEY, NEWS_SOURCES, DIR_NEWS


# ─── Chargement des news existantes ──────────────────────────────────────────

def load_existing_news(company: str, news_dir: str = DIR_NEWS) -> dict:
    """
    Charge le fichier JSON de news déjà scrappées pour une entreprise.

    Args:
        company:  Nom de l'entreprise.
        news_dir: Dossier contenant les fichiers JSON.

    Returns:
        Dictionnaire {date: [articles]} ou dict vide si le fichier n'existe pas.
    """
    path = os.path.join(news_dir, f"{company.replace(' ', '_')}_news.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─── Scraping d'une entreprise ────────────────────────────────────────────────

def get_news_by_date(
    company_name: str,
    api_key: str = NEWS_API_KEY,
    days_back: int = 10,
    news_dir: str = DIR_NEWS,
) -> dict:
    """
    Scrape les actualités d'une entreprise via NewsAPI sur les `days_back` derniers jours,
    déduplique par rapport aux news déjà sauvegardées et met à jour le fichier JSON.

    Filtre : l'article doit mentionner l'entreprise dans son titre ou sa description.

    Args:
        company_name: Nom de l'entreprise (utilisé comme mot-clé de recherche).
        api_key:      Clé API NewsAPI.
        days_back:    Nombre de jours à remonter dans le passé.
        news_dir:     Dossier de sauvegarde des JSON.

    Returns:
        Dictionnaire {date: [articles]} trié chronologiquement.
    """
    os.makedirs(news_dir, exist_ok=True)
    news_dict = load_existing_news(company_name, news_dir)

    last_day  = datetime.today().strftime("%Y-%m-%d")
    first_day = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "sources":  NEWS_SOURCES,
        "q":        company_name,
        "apiKey":   api_key,
        "language": "en",
        "pageSize": 100,
        "from":     first_day,
        "to":       last_day,
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        r = requests.get("https://newsapi.org/v2/everything",
                         params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"  ✘ {company_name} : HTTP {r.status_code}")
            return news_dict

        for article in r.json().get("articles", []):
            title       = article.get("title", "")       or ""
            description = article.get("description", "") or ""
            source      = article.get("source", {}).get("name", "")
            published   = article.get("publishedAt", "")

            # Vérifier que l'entreprise est mentionnée
            kw = company_name.lower()
            if kw not in title.lower() and kw not in description.lower():
                continue

            date = published.split("T")[0] if "T" in published else published
            news_dict.setdefault(date, [])

            # Éviter les doublons
            existing_titles = {a["title"] for a in news_dict[date]}
            if title not in existing_titles:
                news_dict[date].append({
                    "title":       title,
                    "description": description,
                    "source":      source,
                    "publishedAt": published,
                })

        news_dict = dict(sorted(news_dict.items()))

        path = os.path.join(news_dir, f"{company_name.replace(' ', '_')}_news.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(news_dict, f, indent=2, ensure_ascii=False)

        total = sum(len(v) for v in news_dict.values())
        print(f"  ✔ {company_name} : {total} articles au total")

    except Exception as exc:
        print(f"  ✘ {company_name} : {exc}")

    return news_dict


# ─── Scraping de tout l'univers ───────────────────────────────────────────────

def scrape_all_companies_news(
    companies: dict[str, str],
    api_key: str = NEWS_API_KEY,
    days_back: int = 10,
    news_dir: str = DIR_NEWS,
    sleep_seconds: float = 1.0,
) -> dict[str, dict]:
    """
    Scrape les news pour toutes les entreprises de l'univers, avec pause entre requêtes.

    Les noms sont nettoyés avant la recherche (suppression de '.com', '(LVMH)', etc.).

    Args:
        companies:     Dictionnaire {nom: ticker}.
        api_key:       Clé API NewsAPI.
        days_back:     Nombre de jours à remonter.
        news_dir:      Dossier de sauvegarde.
        sleep_seconds: Pause entre deux requêtes (respect du rate-limit).

    Returns:
        Dictionnaire {nom_entreprise: news_dict}.
    """
    all_news: dict[str, dict] = {}

    for company in companies:
        # Nettoyer le nom : enlever suffixes commerciaux et acronymes entre parenthèses
        clean_name = company.split(".")[0].split(" (")[0].strip()
        all_news[clean_name] = get_news_by_date(clean_name, api_key, days_back, news_dir)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print(f"\nScraping terminé — {len(all_news)} entreprises traitées")
    return all_news


def load_all_news(
    companies: dict[str, str],
    news_dir: str = DIR_NEWS,
) -> dict[str, dict]:
    """
    Recharge en mémoire les fichiers JSON de news déjà téléchargés.

    Args:
        companies: Dictionnaire {nom: ticker}.
        news_dir:  Dossier des fichiers JSON.

    Returns:
        Dictionnaire {nom_entreprise: news_dict}.
    """
    all_news: dict[str, dict] = {}
    for company in companies:
        clean_name = company.split(".")[0].split(" (")[0].strip()
        data = load_existing_news(clean_name, news_dir)
        if data:
            all_news[clean_name] = data
    return all_news
