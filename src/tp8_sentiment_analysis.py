"""
TP8 — Analyse de sentiment & corrélation avec les variations de prix.

Fonctions exportées :
    get_texts_timestamps, get_sentiments,
    align_timestamps, plot_comparison, news_sentiments_pipeline
"""

import os
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pytz
import torch
import yfinance as yf
from datetime import datetime

from transformers import BertTokenizer, BertForSequenceClassification

from configs.settings import FINBERT_MODEL_NAME, DIR_MODELS


# ─── Extraction des textes et timestamps ─────────────────────────────────────

def get_texts_timestamps(news_data: dict) -> tuple[list[str], list]:
    """
    Extrait les textes et timestamps (timezone New York) depuis un dictionnaire de news.

    Les timestamps sont arrondis à l'heure pleine précédente.

    Args:
        news_data: Dictionnaire {date: [articles]} issu du scraping TP6.

    Returns:
        (news_texts, news_timestamps) — deux listes parallèles.
    """
    ny_tz = pytz.timezone("America/New_York")
    texts, timestamps = [], []

    for articles in news_data.values():
        for art in articles:
            pub   = art.get("publishedAt", "")
            title = art.get("title", "")       or ""
            desc  = art.get("description", "") or ""

            try:
                dt_utc = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ")
                dt_utc = pytz.utc.localize(dt_utc)
                dt_ny  = dt_utc.astimezone(ny_tz).replace(minute=0, second=0, microsecond=0)
            except Exception:
                continue

            texts.append(f"{title}. {desc}".strip())
            timestamps.append(dt_ny)

    return texts, timestamps


# ─── Prédiction de sentiment ──────────────────────────────────────────────────

def get_sentiments(model_path: str, texts: list[str]) -> list[int]:
    """
    Prédit le sentiment (0=négatif, 1=neutre, 2=positif) pour chaque texte.

    Utilise le tokenizer de FinBERT et les poids du modèle fine-tuné.

    Args:
        model_path: Chemin du modèle fine-tuné (dossier contenant config.json).
        texts:      Liste de textes à classifier.

    Returns:
        Liste d'entiers {0, 1, 2}.
    """
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    sentiments = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        sentiments.append(torch.argmax(logits, dim=-1).item())
    return sentiments


# ─── Alignement des timestamps sur les heures de marché ──────────────────────

def align_timestamps(timestamps: list) -> list:
    """
    Aligne les timestamps des news sur les horaires d'ouverture du NYSE.

    Règles :
        - 9h30–15h00  → heure de publication conservée
        - 15h00–00h00 → 15h00 le même jour
        - 00h00–9h30  → 15h00 la veille

    Args:
        timestamps: Liste de datetime (timezone-aware ou naive).

    Returns:
        Liste de datetime alignés.
    """
    aligned = []
    for ts in timestamps:
        open_  = ts.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_ = ts.replace(hour=15, minute=0,  second=0, microsecond=0)

        if open_ <= ts < close_:
            aligned.append(ts)
        elif ts >= close_:
            aligned.append(close_)
        else:
            prev = ts - timedelta(days=1)
            aligned.append(prev.replace(hour=15, minute=0, second=0, microsecond=0))
    return aligned


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_comparison(
    df: pd.DataFrame,
    sentiments_a: list[int],
    sentiments_b: list[int],
    timestamps: list,
    title_a: str,
    title_b: str,
    output_dir: str = "outputs/sentiment",
    company_name: str = "",
) -> None:
    """
    Affiche deux graphiques côte à côte : cours de l'action + sentiments prédits.

    Chaque news est représentée par un point coloré superposé sur le cours :
        🟢 Vert = positif | 🟡 Or = neutre | 🔴 Rouge = négatif

    Args:
        df:           DataFrame {Datetime, Close} à intervalles 60 min.
        sentiments_a: Prédictions du modèle A.
        sentiments_b: Prédictions du modèle B.
        timestamps:   Timestamps des articles (tz-naive de préférence).
        title_a:      Titre du sous-graphique A.
        title_b:      Titre du sous-graphique B.
        output_dir:   Dossier de sauvegarde.
        company_name: Utilisé pour le nom du fichier sauvegardé.
    """
    aligned = align_timestamps(timestamps)
    color_map = {0: "red", 1: "gold", 2: "green"}

    sent_map_a: dict = defaultdict(list)
    sent_map_b: dict = defaultdict(list)
    for ts, sa, sb in zip(aligned, sentiments_a, sentiments_b):
        sent_map_a[ts].append(sa)
        sent_map_b[ts].append(sb)

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

    for ax, sent_map, title in [(axes[0], sent_map_a, title_a),
                                 (axes[1], sent_map_b, title_b)]:
        datetimes = pd.to_datetime(df["Datetime"])
        ax.plot(datetimes, df["Close"], color="steelblue", linewidth=1.5, label="Prix")

        price_range = df["Close"].max() - df["Close"].min()
        for ts, sents in sent_map.items():
            mask = datetimes >= pd.Timestamp(ts)
            if not mask.any():
                continue
            close_val = df.loc[mask, "Close"].iloc[0]
            for j, s in enumerate(sents):
                ax.scatter(ts, close_val + j * price_range * 0.01,
                           color=color_map[s], s=50, zorder=5, alpha=0.8)

        legend_elements = [
            Line2D([0], [0], color="steelblue", lw=2, label="Prix clôture"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
                   markersize=8, label="Positif"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",
                   markersize=8, label="Neutre"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=8, label="Négatif"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Prix ($)")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    safe = company_name.replace(" ", "_") if company_name else "comparison"
    plt.savefig(os.path.join(output_dir, f"sentiment_{safe}.png"), dpi=100)
    plt.show()


# ─── Pipeline complet ─────────────────────────────────────────────────────────

def news_sentiments_pipeline(
    all_news: dict[str, dict],
    companies: dict[str, str],
    model_path_a: str,
    model_path_b: str,
    label_a: str = "FinBERT base",
    label_b: str = "FinBERT fine-tuné",
    min_news: int = 25,
    output_dir: str = "outputs/sentiment",
) -> None:
    """
    Pour chaque entreprise ayant suffisamment de news :
        1. Récupère les cours horaires depuis 2025-01-01.
        2. Extrait les textes et timestamps.
        3. Prédit les sentiments avec deux modèles.
        4. Affiche le graphique comparatif.

    Args:
        all_news:    Dictionnaire {nom: news_dict} issu du scraping TP6.
        companies:   Dictionnaire {nom: ticker}.
        model_path_a: Chemin ou identifiant HuggingFace du modèle A.
        model_path_b: Chemin du modèle B fine-tuné.
        label_a:     Titre du sous-graphique A.
        label_b:     Titre du sous-graphique B.
        min_news:    Nombre minimum d'articles requis pour analyser une entreprise.
        output_dir:  Dossier de sauvegarde des graphiques.
    """
    eligible = {
        name: data
        for name, data in all_news.items()
        if sum(len(v) for v in data.values()) >= min_news
    }
    print(f"Entreprises analysées ({len(eligible)}) : {list(eligible.keys())}")

    for company_name, news_data in eligible.items():
        print(f"\n{'='*60}\n  Analyse de sentiment — {company_name}\n{'='*60}")

        symbol = companies.get(company_name)
        if symbol is None:
            print(f"  ✘ Symbole introuvable pour {company_name}")
            continue

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start="2025-01-01", interval="60m").reset_index()
            df = df[["Datetime", "Close"]]
            df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
        except Exception as exc:
            print(f"  ✘ Cours introuvables pour {company_name} : {exc}")
            continue

        texts, timestamps = get_texts_timestamps(news_data)
        if not texts:
            print(f"  ✘ Aucun texte extrait")
            continue

        # Rendre les timestamps tz-naive pour la comparaison avec df["Datetime"]
        timestamps = [ts.replace(tzinfo=None) for ts in timestamps]
        print(f"  ✔ {len(texts)} articles")

        print(f"  → Inférence {label_a}...")
        sentiments_a = get_sentiments(model_path_a, texts)

        if os.path.exists(model_path_b):
            print(f"  → Inférence {label_b}...")
            sentiments_b = get_sentiments(model_path_b, texts)
        else:
            print(f"  ⚠ Modèle B non trouvé — utilisation du modèle A en fallback")
            sentiments_b = sentiments_a
            label_b = label_a

        plot_comparison(
            df, sentiments_a, sentiments_b, timestamps,
            f"{label_a} — {company_name}", f"{label_b} — {company_name}",
            output_dir=output_dir, company_name=company_name,
        )
