"""
TP4 — Régression ML pour la prédiction de prix à J+1.

Modèles :  XGBoost, Random Forest, KNN, Ridge
Stratégie : GridSearch sur un représentant par cluster,
            puis entraînement sur toutes les entreprises du cluster.

Fonctions exportées :
    create_target_features, prepare_regression_data,
    find_best_regression_params, run_all_regressors,
    get_cluster_representatives, run_regression_pipeline,
    plot_regression_predictions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from configs.settings import N_DAYS

TSCV = TimeSeriesSplit(n_splits=5, gap=1)


# ─── Préparation des données ─────────────────────────────────────────────────

def create_target_features(
    data: np.ndarray,
    n: int = N_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforme une série temporelle 1D en features glissantes (X) et cibles (y).

    Pour chaque point i ≥ n :
        X[i] = data[i-n : i]   (les n jours précédents)
        y[i] = data[i]          (le jour courant = J+1 à prédire)

    Args:
        data: Tableau 2D (n_samples, 1) normalisé.
        n:    Taille de la fenêtre glissante.

    Returns:
        (X array shape (m, n), y array shape (m,))
    """
    x, y = [], []
    for i in range(n, data.shape[0]):
        x.append(data[i - n:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


def prepare_regression_data(
    df: pd.DataFrame,
    n: int = N_DAYS,
) -> dict:
    """
    Normalise les prix de clôture et crée les datasets train/test fenêtrés.

    Le scaler est fitté uniquement sur le train pour éviter le data leakage.
    Le test inclut les n derniers jours du train comme contexte initial.

    Args:
        df: DataFrame avec colonne 'Close'.
        n:  Taille de la fenêtre glissante.

    Returns:
        Dictionnaire avec clés : X_train, y_train, X_test, y_test, scaler.
    """
    close = df[["Close"]].values
    split_idx = int(len(close) * 0.8)

    train_raw = close[:split_idx]
    test_raw  = close[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    # On préfixe le test avec les n derniers points du train pour ne pas perdre de contexte
    test_context = np.vstack([train_raw[-n:], test_raw])
    test_scaled  = scaler.transform(test_context)

    X_train, y_train = create_target_features(train_scaled, n)
    X_test,  y_test  = create_target_features(test_scaled,  n)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
    }


# ─── GridSearch ───────────────────────────────────────────────────────────────

def find_best_regression_params(reg_data: dict) -> dict[str, dict]:
    """
    Effectue un GridSearch pour chaque régresseur sur les données fournies.

    Args:
        reg_data: Dictionnaire retourné par prepare_regression_data.

    Returns:
        Dictionnaire {nom_modèle: meilleurs_hyperparamètres}.
    """
    searches = [
        (
            XGBRegressor(),
            {"n_estimators": [100, 200, 300], "max_depth": [2, 3, 4],
             "learning_rate": [0.01, 0.05, 0.1]},
            "XGBoost",
        ),
        (
            RandomForestRegressor(),
            {"n_estimators": [100, 200], "min_samples_leaf": [1, 5],
             "max_features": ["sqrt", 0.5]},
            "Random Forest",
        ),
        (
            KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]},
            "KNN",
        ),
        (
            Ridge(),
            {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "Ridge",
        ),
    ]

    best_params: dict[str, dict] = {}
    for model, param_grid, name in searches:
        grid = GridSearchCV(model, param_grid, cv=TSCV,
                            scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(reg_data["X_train"], reg_data["y_train"])
        best_params[name] = grid.best_params_
        print(f"  {name} → {best_params[name]}")

    return best_params


# ─── Entraînement et évaluation ───────────────────────────────────────────────

def run_all_regressors(
    reg_data: dict,
    company: str,
    best_params: dict[str, dict],
) -> list[dict]:
    """
    Entraîne les quatre régresseurs avec les hyperparamètres fournis et évalue leurs performances.

    Args:
        reg_data:    Données de régression (retour de prepare_regression_data).
        company:     Nom de l'entreprise (pour les logs).
        best_params: Dictionnaire {nom: hyperparamètres} retourné par find_best_regression_params.

    Returns:
        Liste de dictionnaires {name, MAE, MSE, RMSE, MAPE, y_pred, y_real, model}.
    """
    model_configs = [
        (XGBRegressor(**best_params.get("XGBoost", {})),    "XGBoost"),
        (RandomForestRegressor(**best_params.get("Random Forest", {})), "Random Forest"),
        (KNeighborsRegressor(**best_params.get("KNN", {})), "KNN"),
        (Ridge(**best_params.get("Ridge", {})),             "Ridge"),
    ]

    results = []
    for model, name in model_configs:
        model.fit(reg_data["X_train"], reg_data["y_train"])
        scaler = reg_data["scaler"]

        y_pred = scaler.inverse_transform(
            model.predict(reg_data["X_test"]).reshape(-1, 1)
        ).flatten()
        y_real = scaler.inverse_transform(
            reg_data["y_test"].reshape(-1, 1)
        ).flatten()

        mae  = mean_absolute_error(y_real, y_pred)
        mse  = mean_squared_error(y_real, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_real, y_pred) * 100

        print(f"  {name} [{company}] — MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
        results.append({
            "name": name, "MAE": round(mae, 2), "MSE": round(mse, 2),
            "RMSE": round(rmse, 2), "MAPE": round(mape, 2),
            "y_pred": y_pred, "y_real": y_real, "model": model,
        })
    return results


# ─── Pipeline complet ─────────────────────────────────────────────────────────

def get_cluster_representatives(
    company_to_cluster: dict[str, int],
    historical_data: dict[str, pd.DataFrame],
) -> dict[int, str]:
    """
    Sélectionne le représentant de chaque cluster (entreprise avec le plus de données).

    Args:
        company_to_cluster: Dictionnaire {nom_entreprise: cluster_id}.
        historical_data:    Données historiques disponibles.

    Returns:
        Dictionnaire {cluster_id: nom_représentant}.
    """
    clusters: dict[int, list[str]] = {}
    for company, cluster in company_to_cluster.items():
        if company in historical_data:
            clusters.setdefault(cluster, []).append(company)

    return {
        cid: max(lst, key=lambda c: len(historical_data[c]))
        for cid, lst in clusters.items()
    }


def run_regression_pipeline(
    companies: dict[str, str],
    historical_data: dict[str, pd.DataFrame],
    company_to_cluster: dict[str, int],
    output_dir: str = "outputs/regression",
) -> tuple[dict, pd.DataFrame]:
    """
    Pipeline complet :
        1. GridSearch sur le représentant de chaque cluster.
        2. Entraînement sur toutes les entreprises du cluster.
        3. Export des métriques.

    Args:
        companies:          Dictionnaire {nom: ticker}.
        historical_data:    Données historiques.
        company_to_cluster: Appartenance de chaque entreprise à un cluster.
        output_dir:         Dossier de sauvegarde.

    Returns:
        (all_ml_results dict, perf_df DataFrame).
    """
    os.makedirs(output_dir, exist_ok=True)
    cluster_reps = get_cluster_representatives(company_to_cluster, historical_data)
    print("Représentants par cluster :", cluster_reps)

    # GridSearch sur chaque représentant
    best_params_per_cluster: dict[int, dict] = {}
    for cid, rep in cluster_reps.items():
        print(f"\n=== GridSearch cluster {cid} → {rep} ===")
        best_params_per_cluster[cid] = find_best_regression_params(
            prepare_regression_data(historical_data[rep])
        )

    all_ml_results: dict[str, list] = {}
    perf_rows: list[dict] = []

    for company in companies:
        if company not in historical_data or company not in company_to_cluster:
            continue
        print(f"\n{'='*50}\n  {company}\n{'='*50}")
        try:
            cid        = company_to_cluster[company]
            reg_data   = prepare_regression_data(historical_data[company])
            results    = run_all_regressors(reg_data, company, best_params_per_cluster[cid])
            all_ml_results[company] = results
            for r in results:
                perf_rows.append({
                    "Entreprise": company, "Cluster": cid,
                    "Modèle": r["name"], "MAE": r["MAE"],
                    "RMSE": r["RMSE"], "MAPE": r["MAPE"],
                })
        except Exception as exc:
            print(f"  ✘ {company} : {exc}")

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(os.path.join(output_dir, "regression_performance.csv"), index=False)
    print("\n=== Métriques de régression exportées ===")
    print(perf_df.groupby("Modèle")[["MAE", "RMSE", "MAPE"]].mean().round(2))
    return all_ml_results, perf_df


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_regression_predictions(
    all_ml_results: dict,
    cluster_reps: dict[int, str],
    output_dir: str = "outputs/regression",
) -> None:
    """
    Trace les prédictions vs valeurs réelles pour le représentant de chaque cluster.

    Args:
        all_ml_results: Dictionnaire {nom_entreprise: liste_résultats}.
        cluster_reps:   Dictionnaire {cluster_id: nom_représentant}.
        output_dir:     Dossier de sauvegarde des graphiques.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = ["blue", "orange", "green", "purple"]

    for cid, company in cluster_reps.items():
        if company not in all_ml_results:
            continue
        results = all_ml_results[company]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(results[0]["y_real"], color="black", label="Valeurs réelles", linewidth=2)
        for res, col in zip(results, colors):
            ax.plot(res["y_pred"], color=col, label=res["name"], alpha=0.7)
        ax.set_title(f"Prédictions J+1 — {company} (Cluster {cid})")
        ax.set_xlabel("Jours")
        ax.set_ylabel("Prix ($)")
        ax.legend()
        plt.tight_layout()
        safe = company.replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"predictions_{safe}.png"), dpi=100)
        plt.show()
