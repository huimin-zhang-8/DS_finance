"""
Configuration globale du projet — paramètres partagés entre tous les modules.
"""
from datetime import datetime, timedelta
import os

# ─── Clés API ────────────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "8df5fe70ce6c47f799dbb257de5c2821")

# ─── Fenêtre temporelle ───────────────────────────────────────────────────────
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

# ─── Répertoires ─────────────────────────────────────────────────────────────
DIR_HISTORICAL = "data/raw/companies_historical_data"
DIR_NEWS       = "data/raw/news_data"
DIR_MODELS     = "models"
DIR_OUTPUTS    = "outputs"

# ─── Univers d'entreprises ───────────────────────────────────────────────────
COMPANIES: dict[str, str] = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Samsung": "005930.KS",
    "Tencent": "TCEHY",
    "Alibaba": "BABA",
    "IBM": "IBM",
    "Intel": "INTC",
    "Oracle": "ORCL",
    "Sony": "SONY",
    "Adobe": "ADBE",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Qualcomm": "QCOM",
    "Cisco": "CSCO",
    "JP Morgan": "JPM",
    "Goldman Sachs": "GS",
    "Visa": "V",
    "Johnson & Johnson": "JNJ",
    "Pfizer": "PFE",
    "ExxonMobil": "XOM",
    "ASML": "ASML.AS",
    "SAP": "SAP.DE",
    "Siemens": "SIE.DE",
    "Louis Vuitton": "MC.PA",
    "TotalEnergies": "TTE.PA",
    "Shell": "SHEL.L",
    "Baidu": "BIDU",
    "JD": "JD",
    "BYD": "BYDDY",
    "ICBC": "1398.HK",
    "Toyota": "TM",
    "SoftBank": "9984.T",
    "Nintendo": "NTDOY",
    "Hyundai": "005380.KS",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
}

# ─── Ratios financiers ────────────────────────────────────────────────────────
RATIOS: list[str] = [
    "forwardPE", "beta", "priceToBook", "priceToSales",
    "dividendYield", "trailingEps", "debtToEquity",
    "currentRatio", "quickRatio", "returnOnEquity",
    "returnOnAssets", "operatingMargins", "profitMargins",
]

# ─── Features de clustering ───────────────────────────────────────────────────
FINANCIAL_FEATURES: list[str] = [
    "forwardPE", "beta", "priceToBook",
    "returnOnEquity", "profitMargins", "operatingMargins",
]

RISK_FEATURES: list[str] = [
    "beta", "debtToEquity", "currentRatio",
    "quickRatio", "returnOnAssets", "dividendYield",
]

# ─── Régression ───────────────────────────────────────────────────────────────
N_DAYS = 30  # fenêtre glissante pour la régression

# ─── Sources NewsAPI ──────────────────────────────────────────────────────────
NEWS_SOURCES = (
    "financial-post,the-wall-street-journal,bloomberg,"
    "the-washington-post,australian-financial-review,bbc-news,cnn"
)

# ─── Modèles NLP ─────────────────────────────────────────────────────────────
BERT_MODEL_NAME    = "bert-base-uncased"
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# ─── Stratégie d'agrégation ───────────────────────────────────────────────────
WEIGHT_CLASSIFICATION = 0.40
WEIGHT_REGRESSION     = 0.35
WEIGHT_SENTIMENT      = 0.25
THRESHOLD_BUY         = 0.15
THRESHOLD_SELL        = -0.15
