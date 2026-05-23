"""
main.py — Pipeline quotidien d'analyse des marchés financiers.

Orchestre les modules TP1→TP8 dans l'ordre et produit :
    - recommendations_{date}.csv  : conseil BUY/HOLD/SELL par entreprise
    - Dashboard visuel des recommandations

Usage :
    python main.py [--skip-scraping] [--skip-training] [--demo N]

Flags :
    --skip-scraping   Réutilise les données déjà téléchargées (CSV + JSON)
    --skip-training   Réutilise les modèles déjà entraînés
    --demo N          Limite le pipeline aux N premières entreprises
"""

import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────
from configs.settings import (
    COMPANIES, RATIOS, START_DATE, END_DATE,
    FINANCIAL_FEATURES, RISK_FEATURES, N_DAYS,
    NEWS_API_KEY, FINBERT_MODEL_NAME, DIR_MODELS,
    WEIGHT_CLASSIFICATION, WEIGHT_REGRESSION, WEIGHT_SENTIMENT,
    THRESHOLD_BUY, THRESHOLD_SELL,
    DIR_HISTORICAL, DIR_NEWS,
)

# ─── TP modules ──────────────────────────────────────────────────────────────
from src.tp1_scraping import (
    scrape_financial_ratios, clean_financial_ratios,
    scrape_stock_history, load_historical_data,
)
from src.tp2_clustering import (
    preprocess_for_clustering, elbow_method, do_kmeans_clustering,
    do_hierarchical_clustering, prepare_daily_returns_df,
    cluster_by_returns_correlation, compare_clustering_algorithms,
)
from src.tp3_classification import (
    create_classification_labels, add_technical_indicators,
    build_classification_dataset, run_all_classifiers,
    build_classification_comparison_table,
)
from src.tp4_regression import (
    prepare_regression_data, run_regression_pipeline,
    get_cluster_representatives,
)
from src.tp5_deep_learning import run_deep_learning_comparison
from src.tp6_news_scraping import scrape_all_companies_news, load_all_news
from src.tp7_bert_finetuning import load_and_merge_datasets, finetune_model, compare_bert_results
from src.tp8_sentiment_analysis import (
    get_texts_timestamps, get_sentiments,
    news_sentiments_pipeline,
)

# ─── Initialisation ───────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor, XGBClassifier


def setup_directories() -> None:
    """Crée tous les dossiers nécessaires au pipeline."""
    for path in [
        "data/raw/companies_historical_data",
        "data/raw/news_data",
        "data/processed",
        "models",
        "outputs/clustering",
        "outputs/classification",
        "outputs/regression",
        "outputs/sentiment",
        "outputs/recommendations",
    ]:
        os.makedirs(path, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Scraping des données (TP1)
# ══════════════════════════════════════════════════════════════════════════════

def step_scraping(companies: dict, skip: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Scrape ou recharge les ratios financiers et l'historique de prix.

    Args:
        companies: Univers d'entreprises.
        skip:      Si True, charge depuis les fichiers existants.

    Returns:
        (financial_ratios_df, historical_data)
    """
    print("\n" + "="*60)
    print("ÉTAPE 1 — Scraping des données (TP1)")
    print("="*60)

    ratios_path = "data/processed/financial_ratios.csv"

    if skip and os.path.exists(ratios_path):
        financial_ratios_df = pd.read_csv(ratios_path, index_col="Company")
        print(f"  ✔ Ratios chargés depuis {ratios_path}")
    else:
        financial_ratios_df = scrape_financial_ratios(companies, RATIOS, ratios_path)
        financial_ratios_df = clean_financial_ratios(financial_ratios_df, ratios_path)

    if skip and os.path.exists(DIR_HISTORICAL):
        historical_data = load_historical_data(DIR_HISTORICAL)
        print(f"  ✔ Historique chargé pour {len(historical_data)} entreprises")
    else:
        historical_data = scrape_stock_history(companies, START_DATE, END_DATE, DIR_HISTORICAL)

    return financial_ratios_df, historical_data


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Clustering (TP2)
# ══════════════════════════════════════════════════════════════════════════════

def step_clustering(
    financial_ratios_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Applique les algorithmes de clustering et retourne les résultats.

    Args:
        financial_ratios_df: Ratios financiers nettoyés.

    Returns:
        (silhouette_df, kmeans_financial_df, company_to_cluster dict)
    """
    print("\n" + "="*60)
    print("ÉTAPE 2 — Clustering (TP2)")
    print("="*60)

    silhouette_df, _ = compare_clustering_algorithms(
        financial_ratios_df, 
        output_dir="outputs/clustering",
    )

    # K-Means sur profils financiers (utilisé par la suite pour la régression)
    fin_scaled, fin_labels = preprocess_for_clustering(financial_ratios_df, FINANCIAL_FEATURES)
    k_fin = elbow_method(fin_scaled, title="Financial Profiles",
                          output_dir="outputs/clustering")
    kmeans_df = do_kmeans_clustering(fin_scaled, fin_labels, k_fin,
                                      title="Financial Profiles",
                                      output_dir="outputs/clustering")
    # Index entreprises sur la colonne cluster
    kmeans_df.index = fin_labels
    company_to_cluster = kmeans_df["cluster"].to_dict()

    return silhouette_df, kmeans_df, company_to_cluster


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Classification Buy/Hold/Sell (TP3)
# ══════════════════════════════════════════════════════════════════════════════

def step_classification(
    historical_data: dict,
) -> tuple[object, StandardScaler, list[str], list[dict]]:
    """
    Entraîne les classifieurs et retourne le meilleur modèle.

    Args:
        historical_data: Données historiques des entreprises.

    Returns:
        (best_model, scaler, feature_names, all_results)
    """
    print("\n" + "="*60)
    print("ÉTAPE 3 — Classification Buy/Hold/Sell (TP3)")
    print("="*60)

    X_train, X_test, y_train, y_test, X_df = build_classification_dataset(historical_data)
    results = run_all_classifiers(X_train, y_train, X_test, y_test)
    build_classification_comparison_table(
        baseline_results=results,
        X_train=X_train,
        X_test=X_test,
        feature_names=X_df.columns.tolist(),
        output_dir="outputs/classification",
    )

    best = max(results, key=lambda r: r["f1_macro"])
    print(f"\n  → Meilleur classifieur : {best['name']} (F1={best['f1_macro']})")

    # Re-fitter le scaler sur toutes les données pour l'inférence en production
    scaler = StandardScaler()
    frames = []
    for df in historical_data.values():
        try:
            labeled   = create_classification_labels(df)
            with_tech = add_technical_indicators(labeled)
            with_tech.dropna(inplace=True)
            drop = [c for c in ["Label", "Close Horizon", "horizon_return",
                                  "Next Day Close", "Daily Return"] if c in with_tech.columns]
            frames.append(with_tech.drop(columns=drop))
        except Exception:
            pass
    if frames:
        X_all = pd.concat(frames)
        scaler.fit(X_all)
        feature_names = X_all.columns.tolist()
    else:
        feature_names = X_df.columns.tolist()

    return best["model"], scaler, feature_names, results


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4+5 — Régression ML + Deep Learning
# ══════════════════════════════════════════════════════════════════════════════

def step_regression(
    companies: dict,
    historical_data: dict,
    company_to_cluster: dict,
) -> tuple[object, dict]:
    """
    Entraîne les régresseurs ML et DL, retourne le meilleur modèle ML.

    Args:
        companies:          Univers d'entreprises.
        historical_data:    Données historiques.
        company_to_cluster: Appartenance de chaque entreprise à un cluster.

    Returns:
        (best_reg_model, all_ml_results)
    """
    print("\n" + "="*60)
    print("ÉTAPE 4 — Régression ML (TP4)")
    print("="*60)

    all_ml_results, perf_df = run_regression_pipeline(
        companies, historical_data, company_to_cluster,
        output_dir="outputs/regression",
    )

    # Deep Learning sur un représentant (Apple ou première entreprise disponible)
    print("\n" + "="*60)
    print("ÉTAPE 5 — Réseaux de neurones (TP5)")
    print("="*60)

    demo_company = next((c for c in ["Apple", "Microsoft"] if c in historical_data), None)
    if demo_company:
        reg_data = prepare_regression_data(historical_data[demo_company])
        dl_models, dl_perf = run_deep_learning_comparison(
            reg_data, output_dir="outputs/regression"
        )
        print(f"\n  Deep Learning sur {demo_company} :")
        print(dl_perf.to_string(index=False))

    # Modèle de régression final : XGBoost du meilleur cluster
    best_reg = None
    if all_ml_results:
        first_company = next(iter(all_ml_results))
        for r in all_ml_results[first_company]:
            if r["name"] == "XGBoost":
                best_reg = r["model"]
                break
    if best_reg is None:
        reg_data = prepare_regression_data(
            historical_data[next(iter(historical_data))]
        )
        best_reg = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        best_reg.fit(reg_data["X_train"], reg_data["y_train"])

    return best_reg, all_ml_results


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Scraping des news (TP6)
# ══════════════════════════════════════════════════════════════════════════════

def step_news_scraping(companies: dict, skip: bool = False) -> dict:
    """
    Scrape ou recharge les news financières.

    Args:
        companies: Univers d'entreprises.
        skip:      Si True, charge les JSON existants.

    Returns:
        Dictionnaire {nom_entreprise: news_dict}.
    """
    print("\n" + "="*60)
    print("ÉTAPE 6 — Scraping des news (TP6)")
    print("="*60)

    if skip:
        all_news = load_all_news(companies, DIR_NEWS)
        print(f"  ✔ News chargées pour {len(all_news)} entreprises")
    else:
        all_news = scrape_all_companies_news(
            companies, api_key=NEWS_API_KEY, days_back=10, news_dir=DIR_NEWS
        )
    return all_news


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Fine-tuning BERT / FinBERT (TP7)
# ══════════════════════════════════════════════════════════════════════════════

def step_bert_finetuning(skip: bool = False) -> tuple[str, str]:
    """
    Fine-tune BERT et FinBERT ou recharge les modèles sauvegardés.

    Args:
        skip: Si True, vérifie si les modèles existent déjà.

    Returns:
        (bert_model_path, finbert_model_path)
    """
    print("\n" + "="*60)
    print("ÉTAPE 7 — Fine-tuning BERT / FinBERT (TP7)")
    print("="*60)

    bert_path    = os.path.join(DIR_MODELS, "best_model_bert-base-uncased")
    finbert_path = os.path.join(DIR_MODELS, "best_model_ProsusAI_finbert")

    if skip or (os.path.exists(bert_path) and os.path.exists(finbert_path)):
        print("  ✔ Modèles fine-tunés déjà disponibles")
        return bert_path, finbert_path

    merged_dataset = load_and_merge_datasets()

    bert_metrics, bert_path = finetune_model(
        "bert-base-uncased", merged_dataset, batch_size=16, num_epochs=2,
        output_dir=DIR_MODELS,
    )
    finbert_metrics, finbert_path = finetune_model(
        FINBERT_MODEL_NAME, merged_dataset, batch_size=16, num_epochs=2,
        output_dir=DIR_MODELS,
    )
    compare_bert_results(bert_metrics, finbert_metrics, output_dir="outputs/sentiment")

    return bert_path, finbert_path


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 8 — Analyse de sentiment + visualisation (TP8)
# ══════════════════════════════════════════════════════════════════════════════

def step_sentiment_analysis(
    all_news: dict,
    companies: dict,
    finbert_base_path: str,
    finbert_finetuned_path: str,
) -> None:
    """
    Lance le pipeline de visualisation des sentiments vs cours boursiers.

    Args:
        all_news:              News scrappées.
        companies:             Univers d'entreprises.
        finbert_base_path:     Identifiant ou chemin de FinBERT base.
        finbert_finetuned_path: Chemin du FinBERT fine-tuné.
    """
    print("\n" + "="*60)
    print("ÉTAPE 8 — Analyse de sentiment (TP8)")
    print("="*60)

    news_sentiments_pipeline(
        all_news=all_news,
        companies=companies,
        model_path_a=finbert_base_path,
        model_path_b=finbert_finetuned_path,
        label_a="FinBERT base",
        label_b="FinBERT fine-tuné",
        min_news=25,
        output_dir="outputs/sentiment",
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGRÉGATION DES SIGNAUX & RECOMMANDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_sentiment_signal(
    news_data: dict,
    model_path: str,
    max_articles: int = 20,
) -> float:
    """
    Calcule un score de sentiment agrégé [-1, +1] à partir des dernières news.

    Mapping : 0 (négatif)→-1 | 1 (neutre)→0 | 2 (positif)→+1

    Args:
        news_data:    Dictionnaire {date: [articles]}.
        model_path:   Chemin du modèle de sentiment.
        max_articles: Nombre maximum d'articles récents à analyser.

    Returns:
        Score flottant entre -1 et +1.
    """
    texts, _ = get_texts_timestamps(news_data)
    if not texts:
        return 0.0
    sentiments = get_sentiments(model_path, texts[-max_articles:])
    return round(float(np.mean([s - 1 for s in sentiments])), 3)


def aggregate_signals(
    company_name: str,
    symbol: str,
    cls_model,
    reg_model,
    kmeans_df: pd.DataFrame,
    news_data: dict,
    sentiment_model_path: str,
    cls_scaler: StandardScaler,
    feature_names: list[str],
    n_days: int = N_DAYS,
) -> dict:
    """
    Agrège les signaux de classification, régression et sentiment en un score final.

    Pondération :
        - Classification (buy/hold/sell) : 40%
        - Régression J+1 (rendement prédit) : 35%
        - Sentiment news : 25%

    Règles de décision :
        - score > +0.15 → BUY
        - score < -0.15 → SELL
        - sinon         → HOLD

    Args:
        company_name:          Nom de l'entreprise.
        symbol:                Ticker yfinance.
        cls_model:             Modèle de classification entraîné.
        reg_model:             Modèle de régression entraîné.
        kmeans_df:             DataFrame K-Means avec colonne 'cluster'.
        news_data:             News scrappées pour cette entreprise.
        sentiment_model_path:  Chemin du modèle de sentiment.
        cls_scaler:            Scaler StandardScaler fitté sur les données de classification.
        feature_names:         Noms des features de classification.
        n_days:                Fenêtre glissante pour la régression.

    Returns:
        Dictionnaire de résultats avec recommendation, scores, etc.
    """
    import yfinance as yf
    from src.tp3_classification import create_classification_labels, add_technical_indicators

    result = {"company": company_name, "symbol": symbol}

    # ── 1. Données récentes ──────────────────────────────────────────────────
    try:
        raw = yf.download(symbol, period="90d", progress=False)
        if raw.empty:
            raise ValueError("données vides")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close_series = raw[["Close"]].copy()
    except Exception as exc:
        result["error"] = str(exc)
        return result

    # ── 2. Signal de classification ──────────────────────────────────────────
    cls_signal = 0
    try:
        labeled   = create_classification_labels(close_series)
        with_tech = add_technical_indicators(labeled)
        with_tech.dropna(inplace=True)
        drop = [c for c in ["Label", "Close Horizon", "horizon_return",
                               "Next Day Close", "Daily Return"] if c in with_tech.columns]
        X_latest = with_tech.drop(columns=drop).tail(1)
        X_latest = X_latest.reindex(columns=feature_names, fill_value=0)
        X_scaled = cls_scaler.transform(X_latest)
        cls_pred = int(cls_model.predict(X_scaled)[0])
        cls_signal = cls_pred - 1
        result["classification"] = ["Sell", "Hold", "Buy"][cls_pred]
    except Exception as exc:
        result["classification"] = f"Indisponible ({exc})"

    # ── 3. Signal de régression ───────────────────────────────────────────────
    reg_signal = 0
    try:
        sc = MinMaxScaler()
        scaled_close = sc.fit_transform(close_series.values)
        if len(scaled_close) >= n_days:
            x_reg   = scaled_close[-n_days:].reshape(1, -1)
            pred_sc = reg_model.predict(x_reg)[0]
            pred_price  = sc.inverse_transform([[pred_sc]])[0][0]
            last_price  = float(close_series["Close"].iloc[-1])
            pred_return = (pred_price - last_price) / last_price
            reg_signal  = float(np.clip(pred_return * 10, -1, 1))
            result["predicted_price"]  = round(pred_price, 2)
            result["predicted_return"] = round(pred_return * 100, 2)
        else:
            result["predicted_return"] = "Données insuffisantes"
    except Exception as exc:
        result["predicted_return"] = f"Erreur ({exc})"

    # ── 4. Signal de sentiment ────────────────────────────────────────────────
    sent_signal = 0.0
    try:
        sent_signal = compute_sentiment_signal(news_data, sentiment_model_path)
    except Exception:
        pass
    result["sentiment_score"] = sent_signal

    # ── 5. Entreprises similaires (cluster) ───────────────────────────────────
    if company_name in kmeans_df.index:
        c = kmeans_df.loc[company_name, "cluster"]
        similar = [s for s in kmeans_df[kmeans_df["cluster"] == c].index if s != company_name]
        result["similar_companies"] = similar
    else:
        result["similar_companies"] = []

    # ── 6. Score agrégé et recommandation ────────────────────────────────────
    final_score = (
        WEIGHT_CLASSIFICATION * cls_signal
        + WEIGHT_REGRESSION   * reg_signal
        + WEIGHT_SENTIMENT    * sent_signal
    )
    if final_score > THRESHOLD_BUY:
        recommendation = "BUY 🟢"
    elif final_score < THRESHOLD_SELL:
        recommendation = "SELL 🔴"
    else:
        recommendation = "HOLD 🟡"

    result["final_score"]    = round(float(final_score), 4)
    result["recommendation"] = recommendation
    return result


def run_daily_pipeline(
    companies: dict,
    cls_model,
    cls_scaler: StandardScaler,
    cls_feature_names: list[str],
    reg_model,
    kmeans_df: pd.DataFrame,
    all_news: dict,
    sentiment_model_path: str,
    output_dir: str = "outputs/recommendations",
) -> pd.DataFrame:
    """
    Exécute le pipeline quotidien de recommandations pour toutes les entreprises.

    Produit un fichier CSV horodaté avec les recommandations.

    Args:
        companies:           Univers d'entreprises.
        cls_model:           Meilleur modèle de classification.
        cls_scaler:          Scaler de classification.
        cls_feature_names:   Noms des features de classification.
        reg_model:           Meilleur modèle de régression.
        kmeans_df:           Résultats K-Means avec clusters.
        all_news:            News scrappées.
        sentiment_model_path: Chemin du modèle de sentiment.
        output_dir:          Dossier de sauvegarde.

    Returns:
        DataFrame des recommandations.
    """
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  PIPELINE QUOTIDIEN — {today}")
    print(f"{'='*60}\n")

    recommendations = []
    for company_name, symbol in companies.items():
        clean_name = company_name.split(".")[0].split(" (")[0].strip()
        print(f"Analyse de {company_name} ({symbol})...")
        news = all_news.get(clean_name, all_news.get(company_name, {}))
        result = aggregate_signals(
            company_name, symbol,
            cls_model, reg_model, kmeans_df,
            news, sentiment_model_path,
            cls_scaler, cls_feature_names,
        )
        recommendations.append(result)
        print(f"  → {result.get('recommendation', 'N/A')} "
              f"(score={result.get('final_score', 'N/A')})")

    output_df = pd.DataFrame(recommendations)
    out_path = os.path.join(output_dir, f"recommendations_{today}.csv")
    output_df.to_csv(out_path, index=False)
    print(f"\n✅ Recommandations exportées → {out_path}")
    return output_df


def plot_dashboard(
    recommendations_df: pd.DataFrame,
    output_dir: str = "outputs/recommendations",
) -> None:
    """
    Affiche un tableau de bord horizontal des scores agrégés par entreprise.

    Args:
        recommendations_df: DataFrame retourné par run_daily_pipeline.
        output_dir:         Dossier de sauvegarde du graphique.
    """
    df = recommendations_df.copy()
    df = df[pd.to_numeric(df["final_score"], errors="coerce").notna()]
    df["final_score"] = df["final_score"].astype(float)
    df = df.sort_values("final_score", ascending=True)

    colors = [
        "limegreen" if "BUY"  in str(r) else
        "tomato"    if "SELL" in str(r) else
        "gold"
        for r in df["recommendation"]
    ]

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))
    bars = ax.barh(df["company"], df["final_score"], color=colors,
                   edgecolor="white", height=0.6)
    ax.axvline(0,               color="black", linewidth=1)
    ax.axvline(THRESHOLD_BUY,   color="green", linewidth=1, linestyle="--", alpha=0.5, label="Seuil BUY")
    ax.axvline(THRESHOLD_SELL,  color="red",   linewidth=1, linestyle="--", alpha=0.5, label="Seuil SELL")

    for bar, rec in zip(bars, df["recommendation"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                str(rec), va="center", fontsize=9)

    ax.set_xlabel("Score agrégé")
    ax.set_title(f"Recommandations — {datetime.today().strftime('%Y-%m-%d')}")
    ax.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"dashboard_{datetime.today().strftime('%Y-%m-%d')}.png"), dpi=100)
    plt.show()

    print("\n=== Résumé final ===")
    cols = ["company", "recommendation", "final_score", "classification",
            "predicted_return", "sentiment_score"]
    available = [c for c in cols if c in recommendations_df.columns]
    print(recommendations_df[available].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline quotidien d'analyse des marchés")
    parser.add_argument("--skip-scraping",  action="store_true",
                        help="Réutilise les données déjà téléchargées")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Réutilise les modèles déjà entraînés")
    parser.add_argument("--demo", type=int, default=0,
                        help="Limite aux N premières entreprises (0 = toutes)")
    args = parser.parse_args()

    setup_directories()

    # Sous-ensemble pour le mode démo
    companies = COMPANIES
    if args.demo > 0:
        companies = dict(list(COMPANIES.items())[:args.demo])
        print(f"Mode démo : {len(companies)} entreprises")

    # ── TP1 : Scraping ────────────────────────────────────────────────────────
    financial_ratios_df, historical_data = step_scraping(companies, skip=args.skip_scraping)

    # ── TP2 : Clustering ──────────────────────────────────────────────────────
    silhouette_df, kmeans_df, company_to_cluster = step_clustering(financial_ratios_df)

    # ── TP3 : Classification ─────────────────────────────────────────────────
    cls_model, cls_scaler, cls_features, cls_results = step_classification(historical_data)

    # ── TP4 + TP5 : Régression ───────────────────────────────────────────────
    reg_model, all_ml_results = step_regression(
        companies, historical_data, company_to_cluster
    )

    # ── TP6 : News ────────────────────────────────────────────────────────────
    all_news = step_news_scraping(companies, skip=args.skip_scraping)

    # ── TP7 : Fine-tuning BERT ───────────────────────────────────────────────
    bert_path, finbert_path = step_bert_finetuning(skip=args.skip_training)

    # ── TP8 : Analyse de sentiment ───────────────────────────────────────────
    step_sentiment_analysis(all_news, companies, FINBERT_MODEL_NAME, finbert_path)

    # ── Agrégation & Recommandations ─────────────────────────────────────────
    sentiment_path = finbert_path if os.path.exists(finbert_path) else FINBERT_MODEL_NAME
    recommendations_df = run_daily_pipeline(
        companies=companies,
        cls_model=cls_model,
        cls_scaler=cls_scaler,
        cls_feature_names=cls_features,
        reg_model=reg_model,
        kmeans_df=kmeans_df,
        all_news=all_news,
        sentiment_model_path=sentiment_path,
    )
    plot_dashboard(recommendations_df)


if __name__ == "__main__":
    main()
