"""
TP3 - Classification Buy / Hold / Sell
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import shap

import ta
import yfinance as yf

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier

from configs.settings import START_DATE, END_DATE

warnings.filterwarnings("ignore")

TSCV = TimeSeriesSplit(n_splits=5, gap=1)


# ─── Labels ──────────────────────────────────────────────────────────────────

def create_classification_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les labels buy/hold/sell à partir du rendement sur 20 jours.

    Règle :
        - rendement > +5%  → Buy  (2)
        - rendement < -5%  → Sell (0)
        - sinon            → Hold (1)

    Args:
        df: DataFrame contenant au minimum une colonne 'Close'.

    Returns:
        DataFrame avec colonnes supplémentaires 'Close Horizon',
        'horizon_return' et 'Label'.
    """
    df = df[["Close"]].copy()
    df["Close Horizon"] = df["Close"].shift(-20)
    df["horizon_return"] = (df["Close Horizon"] - df["Close"]) / df["Close"]
    df["Label"] = df["horizon_return"].apply(
        lambda r: 2 if r > 0.05 else (0 if r < -0.05 else 1)
    )
    return df.dropna()


# ─── Features techniques ─────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les indicateurs techniques via la librairie ta.

    Indicateurs créés :
        SMA_20, EMA_20, RSI_14, MACD, MACD_Signal,
        Bollinger_High, Bollinger_Low, Rolling_Volatility, ROC_10

    Args:
        df: DataFrame avec colonne 'Close'.

    Returns:
        DataFrame enrichi.
    """
    c = df["Close"]
    df["SMA_20"]             = ta.trend.sma_indicator(c, window=20)
    df["EMA_20"]             = ta.trend.ema_indicator(c, window=20)
    df["RSI_14"]             = ta.momentum.rsi(c, window=14)
    macd                     = ta.trend.MACD(c)
    df["MACD"]               = macd.macd()
    df["MACD_Signal"]        = macd.macd_signal()
    bb                       = ta.volatility.BollingerBands(c, window=20)
    df["Bollinger_High"]     = bb.bollinger_hband()
    df["Bollinger_Low"]      = bb.bollinger_lband()
    df["Rolling_Volatility"] = c.rolling(20).std()
    df["ROC_10"]             = ta.momentum.roc(c, window=10)
    return df


def add_macro_temporal_features(
    df: pd.DataFrame,
    vix_df: pd.DataFrame,
    tnx_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ajoute des variables temporelles (mois, jour, trimestre)
    et macro-économiques (VIX, taux 10 ans US).

    Args:
        df:     DataFrame de features techniques.
        vix_df: Historique du VIX (colonne 'Close').
        tnx_df: Historique du TNX (colonne 'Close').

    Returns:
        DataFrame enrichi.
    """
    df = df.copy()
    idx = pd.to_datetime(df.index)
    df["month"]       = idx.month
    df["day_of_week"] = idx.dayofweek
    df["quarter"]     = idx.quarter

    for series, col in [(vix_df, "VIX_Close"), (tnx_df, "TNX_Close")]:
        if not series.empty:
            s = series["Close"].rename(col)
            s.index = pd.to_datetime(s.index)
            df = df.join(s, how="left")

    for col in ["VIX_Close", "TNX_Close"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df


# ─── Construction des datasets ───────────────────────────────────────────────

_DROP_COLS = ["Label", "Close Horizon", "horizon_return", "Next Day Close", "Daily Return"]


def build_classification_dataset(
    historical_data: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Construit le dataset de classification (baseline) à partir des données historiques.

    Concatène toutes les entreprises, applique les features techniques,
    standardise et sépare train/test.

    Args:
        historical_data: Dictionnaire {nom: DataFrame avec colonne 'Close'}.

    Returns:
        (X_train, X_test, y_train, y_test, X_DataFrame_original)
    """
    frames = []
    for name, df in historical_data.items():
        try:
            labeled   = create_classification_labels(df)
            with_tech = add_technical_indicators(labeled)
            with_tech.dropna(inplace=True)
            frames.append(with_tech)
        except Exception as exc:
            print(f"  ✘ {name}: {exc}")

    full_df  = pd.concat(frames)
    drop     = [c for c in _DROP_COLS if c in full_df.columns]
    X        = full_df.drop(columns=drop)
    y        = full_df["Label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset baseline — {full_df.shape} | Train {X_train.shape} | Test {X_test.shape}")
    return X_train, X_test, y_train, y_test, X, scaler


def build_enhanced_classification_dataset(
    historical_data: dict[str, pd.DataFrame],
    start: str = START_DATE,
    end: str = END_DATE,
    X_cls: pd.DataFrame = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Reconstruit le dataset avec features enrichies (macro + temporelles).

    Télécharge VIX et TNX via yfinance, les joint aux features techniques.

    Args:
        historical_data: Dictionnaire {nom: DataFrame}.
        start:           Date de début pour VIX/TNX.
        end:             Date de fin.

    Returns:
        (X_train, X_test, y_train, y_test, X_DataFrame_original)
    """
    print("Téléchargement des données macro (VIX, TNX)...")
    vix_df = yf.download("^VIX", start=start, end=end, progress=False)[["Close"]]
    tnx_df = yf.download("^TNX", start=start, end=end, progress=False)[["Close"]]
    for d in (vix_df, tnx_df):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    frames = []
    for name, df in historical_data.items():
        try:
            labeled   = create_classification_labels(df)
            with_tech = add_technical_indicators(labeled)
            with_macro = add_macro_temporal_features(with_tech, vix_df, tnx_df)
            with_macro.dropna(inplace=True)
            frames.append(with_macro)
        except Exception as exc:
            print(f"  ✘ {name}: {exc}")

    full_df  = pd.concat(frames)
    drop     = [c for c in _DROP_COLS if c in full_df.columns]
    X        = full_df.drop(columns=drop)
    y        = full_df["Label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset enrichi — {full_df.shape} | Train {X_train.shape} | Test {X_test.shape}")
    print(f"\nNouvelles features ajoutées : {[c for c in X.columns if c not in X_cls.columns]}")
    return X_train, X_test, y_train, y_test, X, scaler


# ─── Gestion du déséquilibre des classes ─────────────────────────────────────

def compute_class_weights(y_train: pd.Series) -> dict[int, float]:
    """
    Calcule les poids de classes pour corriger le déséquilibre (méthode 'balanced').

    Args:
        y_train: Série des labels d'entraînement.

    Returns:
        Dictionnaire {classe: poids}.
    """
    classes = np.array(sorted(y_train.unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, weights))
    label_map = {0: "Sell", 1: "Hold", 2: "Buy"}
    print("Poids par classe :")
    for k, w in cw_dict.items():
        print(f"  {label_map[k]} ({k}) : {Counter(y_train)[k]} samples → poids {w:.3f}")
    return cw_dict


# ─── Entraînement ────────────────────────────────────────────────────────────

def train_classifier(
    model,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    GridSearch + évaluation d'un classifieur.

    Args:
        model:      Instance du modèle sklearn/xgboost.
        param_grid: Grille d'hyperparamètres.
        X_train, y_train: Données d'entraînement.
        X_test,  y_test:  Données de test.
        model_name: Nom affiché dans les logs.

    Returns:
        Dictionnaire {'model', 'accuracy', 'f1_macro', 'name'}.
    """
    grid = GridSearchCV(model, param_grid, cv=TSCV, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    best    = grid.best_estimator_
    y_pred  = best.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'='*50}")
    print(f"  {model_name} — Accuracy : {acc:.4f}  F1-macro : {f1:.4f}")
    print(f"  Meilleurs hyperparamètres : {grid.best_params_}")
    print(classification_report(y_test, y_pred, target_names=["Sell", "Hold", "Buy"]))
    return {"model": best, "accuracy": acc, "f1_macro": round(f1, 4), "name": model_name}


def train_classifier_balanced(
    model,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    Identique à train_classifier mais avec class_weight='balanced' intégré.

    Args:
        Mêmes arguments que train_classifier.

    Returns:
        Dictionnaire {'model', 'accuracy', 'f1_macro', 'name'}.
    """
    grid = GridSearchCV(model, param_grid, cv=TSCV, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    best   = grid.best_estimator_
    y_pred = best.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'='*50}")
    print(f"  {model_name} — Accuracy : {acc:.4f}  F1-macro : {f1:.4f}")
    print(f"  Meilleurs hyperparamètres : {grid.best_params_}")
    print(classification_report(y_test, y_pred, target_names=["Sell", "Hold", "Buy"]))
    return {"model": best, "accuracy": acc, "f1_macro": round(f1, 4), "name": model_name}


def run_all_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict]:
    """
    Entraîne les cinq classifieurs (XGBoost, RF, KNN, LR, SVM) avec GridSearch.

    Args:
        X_train, y_train: Données d'entraînement.
        X_test,  y_test:  Données de test.

    Returns:
        Liste de dictionnaires de résultats.
    """
    configs = [
        (
            XGBClassifier(),
            {"n_estimators": [100, 200, 300, 500], 
             "max_depth": [2, 3, 4, 5],
             "learning_rate": [0.01, 0.05, 0.1, 0.2]},
            "XGBoost",
        ),
        (
            RandomForestClassifier(),
            {"n_estimators": [100, 200, 300], 
             "min_samples_leaf": [1, 5, 10],
             "max_features": ["sqrt", 0.5]},
            "Random Forest",
        ),
        (
            KNeighborsClassifier(n_jobs=-1),
            {"n_neighbors": [3, 5, 10, 15], 
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan", "minkowski"]},
            "KNN",
        ),
        (
            LogisticRegression(max_iter=5000, tol=1e-4),
            {"C": [0.01, 0.1, 1, 10, 100], 
             "penalty": ["l1", "l2", "elasticnet"], 
             "solver": ["lbfgs", "saga", "newton-cholesky"]},
            "Logistic Regression",
        ),
        (
            LinearSVC(max_iter=5000, tol=1e-4),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0], 
             "penalty": ["l1", "l2"]},
            "SVM (Linear)",
        ),
    ]
    results = []
    for model, param_grid, name in configs:
        results.append(
            train_classifier(model, param_grid, X_train, y_train, X_test, y_test, name)
        )
    return results


# ─── SHAP ─────────────────────────────────────────────────────────────────────

def run_shap(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    classes: list[str] = ("Sell", "Hold", "Buy"),
    n_samples: int = 500,
    output_dir: str = "outputs/classification",
) -> None:
    """
    Génère les SHAP summary plots pour expliquer les prédictions du modèle.

    Args:
        model_name:    Nom du modèle (utilisé pour le routing de l'explainer).
        model:         Modèle entraîné.
        X_train:       Données d'entraînement (pour KernelExplainer).
        X_test:        Données de test à expliquer.
        feature_names: Noms des features.
        classes:       Labels des classes.
        n_samples:     Nombre d'exemples de test à utiliser.
        output_dir:    Dossier de sauvegarde des plots.
    """
    print(f"\n=== SHAP — {model_name} ===")
    os.makedirs(output_dir, exist_ok=True)
    try:
        idx    = np.random.choice(X_test.shape[0], size=min(n_samples, X_test.shape[0]), replace=False)
        X_small = X_test[idx]

        if "XGBoost" in model_name:
            model.get_booster().feature_names = None
            explainer = shap.TreeExplainer(model, model_output="raw")
        elif "Forest" in model_name:
            explainer = shap.TreeExplainer(model)
        elif "SVM" in model_name:
            summary   = shap.kmeans(X_train, 5)
            explainer = shap.KernelExplainer(model.predict, summary)
        elif "KNN" in model_name:
            sample    = shap.sample(X_train, 20)
            explainer = shap.KernelExplainer(model.predict, sample)
        else:
            explainer = shap.LinearExplainer(model, X_train)

        shap_values = explainer.shap_values(X_small)
        sv = np.array(shap_values)
        for i, label in enumerate(classes):
            if sv.ndim == 3 and sv.shape[0] == len(classes):
                print(f"SHAP — importance pour '{label}' :")
                shap.summary_plot(sv[i], X_small, feature_names=feature_names, show=False)
            elif sv.ndim == 3:
                shap.summary_plot(sv[:, :, i], X_small, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(sv, X_small, feature_names=feature_names, show=False)
                break
            safe = model_name.replace(" ", "_")
            plt.savefig(os.path.join(output_dir, f"shap_{safe}_{label}.png"), bbox_inches="tight")
            plt.close()
    except Exception as exc:
        print(f"  SHAP indisponible pour {model_name} : {exc}")


# ─── Tableau de comparaison ───────────────────────────────────────────────────

def build_classification_comparison_table(
    baseline_results: list[dict],
    enhanced_results: list[dict] | None = None,
    X_train: np.ndarray = None,
    X_test: np.ndarray = None,
    feature_names: list[str] = None,
    output_dir: str = "outputs/classification",
) -> pd.DataFrame:
    """
    Consolide les performances de tous les classifieurs en un DataFrame.

    Args:
        baseline_results: Résultats du dataset baseline.
        enhanced_results: Résultats du dataset enrichi (optionnel).
        X_train:          Données d'entraînement (pour SHAP KernelExplainer).
        X_test:           Données de test à expliquer.
        feature_names:    Noms des features.
        output_dir:       Dossier de sauvegarde.

    Returns:
        DataFrame trié par F1-macro décroissant.
    """
    rows = []
    for r in baseline_results:
        rows.append({"Modèle": r["name"], "Dataset": "Baseline",
                     "Accuracy": round(r["accuracy"], 4), "F1-macro": r["f1_macro"]})
        
        # SHAP pour chaque modèle baseline
        if X_train is not None and X_test is not None and feature_names is not None:
            run_shap(
                model_name=r["name"],
                model=r["model"],
                X_train=X_train,
                X_test=X_test,
                feature_names=feature_names,
                output_dir=output_dir,
            )

    if enhanced_results:
        for r in enhanced_results:
            rows.append({"Modèle": r["name"], "Dataset": "Enrichi",
                         "Accuracy": round(r["accuracy"], 4), "F1-macro": r["f1_macro"]})

    df = pd.DataFrame(rows).sort_values("F1-macro", ascending=False).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "classification_comparison.csv"), index=False)
    print("\n=== Tableau comparatif — Classification ===")
    print(df.to_string(index=False))
    return df
