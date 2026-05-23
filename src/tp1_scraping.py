"""
TP1 - Scraping des données financières
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

from configs.settings import COMPANIES, RATIOS, START_DATE, END_DATE, DIR_HISTORICAL


# ─── 1.1 Ratios financiers ────────────────────────────────────────────────────

def scrape_financial_ratios(
    companies: dict[str, str] = COMPANIES,
    ratios: list[str] = RATIOS,
    output_path: str = "data/processed/financial_ratios.csv",
) -> pd.DataFrame:
    """
    Scrape les ratios financiers fondamentaux pour chaque entreprise via yfinance.

    Args:
        companies:   Dictionnaire {nom: ticker}.
        ratios:      Liste des ratios à extraire.
        output_path: Chemin de sauvegarde du CSV.

    Returns:
        DataFrame (entreprises × ratios).
    """
    data: dict[str, list] = {ratio: [] for ratio in ratios}
    index: list[str] = []

    for company_name, symbol in companies.items():
        try:
            info = yf.Ticker(symbol).info
            for ratio in ratios:
                data[ratio].append(info.get(ratio, np.nan))
            index.append(company_name)
            print(f"  ✔ {company_name}")
        except Exception as exc:
            print(f"  ✘ {company_name} : {exc}")

    df = pd.DataFrame(data, index=index)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index_label="Company")
    print(f"\nRatios exportés → {output_path}  {df.shape}")
    return df


def clean_financial_ratios(
    df: pd.DataFrame,
    output_path: str = "data/processed/financial_ratios.csv",
) -> pd.DataFrame:
    """
    Supprime les colonnes et lignes entièrement vides, puis ré-exporte.

    Args:
        df:          DataFrame brut issu de scrape_financial_ratios.
        output_path: Chemin de sauvegarde.

    Returns:
        DataFrame nettoyé.
    """
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df.to_csv(output_path, index_label="Company")
    return df


# ─── 1.2 Historique des prix ──────────────────────────────────────────────────

def scrape_stock_history(
    companies: dict[str, str] = COMPANIES,
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: str = DIR_HISTORICAL,
) -> dict[str, pd.DataFrame]:
    """
    Télécharge 5 ans d'historique de prix pour chaque entreprise,
    calcule les rendements journaliers et sauvegarde un CSV par entreprise.

    Colonnes produites :
        - Close          : prix de clôture
        - Next Day Close : prix de clôture du lendemain
        - Daily Return   : rendement journalier (J→J+1)

    Args:
        companies:  Dictionnaire {nom: ticker}.
        start:      Date de début (format YYYY-MM-DD).
        end:        Date de fin   (format YYYY-MM-DD).
        output_dir: Dossier de destination des CSV.

    Returns:
        Dictionnaire {nom_entreprise: DataFrame}.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_data: dict[str, pd.DataFrame] = {}

    for company_name, symbol in companies.items():
        try:
            raw = yf.download(symbol, start=start, end=end, progress=False)
            if raw.empty:
                print(f"  ✘ {company_name} : données vides")
                continue

            df = raw[["Close"]].copy()
            # Aplatir le MultiIndex éventuel
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = ["Close"]

            df["Next Day Close"] = df["Close"].shift(-1)
            df["Daily Return"]   = (df["Next Day Close"] - df["Close"]) / df["Close"]
            df.dropna(inplace=True)

            safe_name = company_name.replace("/", "_").replace(" ", "_")
            path = os.path.join(output_dir, f"{safe_name}.csv")
            df.to_csv(path)
            all_data[company_name] = df
            print(f"  ✔ {company_name} → {path}")
        except Exception as exc:
            print(f"  ✘ {company_name} : {exc}")

    print(f"\n{len(all_data)} entreprises téléchargées")
    return all_data


def load_historical_data(
    directory: str = DIR_HISTORICAL,
) -> dict[str, pd.DataFrame]:
    """
    Recharge en mémoire les CSV d'historique précédemment téléchargés.

    Args:
        directory: Dossier contenant les CSV.

    Returns:
        Dictionnaire {nom_entreprise: DataFrame}.
    """
    import glob

    all_data: dict[str, pd.DataFrame] = {}
    for path in glob.glob(os.path.join(directory, "*.csv")):
        name = os.path.basename(path).replace(".csv", "").replace("_", " ")
        try:
            df = pd.read_csv(path, index_col=0)
            all_data[name] = df
        except Exception as exc:
            print(f"  ✘ {name} : {exc}")
    return all_data
