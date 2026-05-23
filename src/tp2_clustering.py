"""
TP2 - Clustering des profils d'entreprises

Méthodes implémentées :
    - K-Means   
    - Hierarchical
    - DBSCAN
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from configs.settings import FINANCIAL_FEATURES, RISK_FEATURES, DIR_HISTORICAL


# ─── Pré-processing ───────────────────────────────────────────────────────────

def preprocess_for_clustering(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Sélectionne les features, supprime les lignes avec NaN et standardise.

    Args:
        df:       DataFrame des ratios financiers.
        features: Liste des colonnes à utiliser.

    Returns:
        (données standardisées, liste des labels d'entreprises).
    """
    subset = df[features].dropna()
    scaled = StandardScaler().fit_transform(subset)
    return scaled, subset.index.tolist()


# ─── K-Means ─────────────────────────────────────────────────────────────────

def elbow_method(
    data: np.ndarray,
    k_max: int = 10,
    title: str = "",
    output_dir: str = "outputs/clustering",
) -> int:
    """
    Trace la courbe d'inertie et retourne le k optimal (méthode du coude).

    Args:
        data:       Données standardisées.
        k_max:      Nombre maximum de clusters à tester.
        title:      Titre du graphique.
        output_dir: Dossier de sauvegarde du graphique.

    Returns:
        k optimal détecté automatiquement.
    """
    inertias = []
    ks = range(2, k_max + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(data)
        inertias.append(km.inertia_)

    diffs  = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_k = int(ks[np.argmax(diffs2) + 1])

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(list(ks), inertias, "bo-")
    plt.axvline(x=elbow_k, color="red", linestyle="--", label=f"Coude k={elbow_k}")
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Inertie")
    plt.title(f"Méthode du coude {title}")
    plt.legend()
    plt.tight_layout()
    safe = title.replace(" ", "_").replace("—", "").strip("_")
    plt.savefig(os.path.join(output_dir, f"elbow_{safe}.png"), dpi=100)
    plt.show()
    return elbow_k


def do_kmeans_clustering(
    data: np.ndarray,
    labels: list[str],
    k: int,
    title: str = "",
    output_dir: str = "outputs/clustering",
) -> pd.DataFrame:
    """
    Applique K-Means et affiche la projection t-SNE colorée par cluster.

    Args:
        data:       Données standardisées.
        labels:     Noms des entreprises.
        k:          Nombre de clusters.
        title:      Titre du graphique.
        output_dir: Dossier de sauvegarde.

    Returns:
        DataFrame avec colonnes de features + colonne 'cluster'.
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(data)

    result_df = pd.DataFrame(data, index=labels)
    result_df["cluster"] = cluster_labels

    print(f"\n=== KMeans {title} — k={k} ===")
    print(result_df["cluster"].value_counts().sort_index())

    # t-SNE pour visualisation
    perp = min(30, max(2, len(data) - 1))
    tsne_coords = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(data)

    os.makedirs(output_dir, exist_ok=True)
    cmap = plt.cm.get_cmap("tab10", k)
    plt.figure(figsize=(10, 6))
    for c in range(k):
        mask = cluster_labels == c
        plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                    color=cmap(c), label=f"Cluster {c}", s=80)
        for i, txt in enumerate(np.array(labels)[mask]):
            plt.annotate(txt, (tsne_coords[mask][i, 0], tsne_coords[mask][i, 1]),
                         fontsize=7, alpha=0.7)
    plt.title(f"t-SNE — KMeans {title}")
    plt.legend()
    plt.tight_layout()
    safe = title.replace(" ", "_").replace("—", "").strip("_")
    plt.savefig(os.path.join(output_dir, f"kmeans_tsne_{safe}.png"), dpi=100)
    plt.show()
    return result_df


# ─── Clustering Hiérarchique ─────────────────────────────────────────────────

def plot_dendrogram(
    data: np.ndarray,
    labels: list[str],
    method: str = "ward",
    title: str = "",
    output_dir: str = "outputs/clustering",
) -> tuple[np.ndarray, int]:
    """
    Calcule la matrice de liaison et trace le dendrogramme.

    Args:
        data:       Données (matrice de distances condensée ou features).
        labels:     Noms des entreprises.
        method:     Méthode de liaison ('ward', 'complete', etc.).
        title:      Titre du graphique.
        output_dir: Dossier de sauvegarde.

    Returns:
        (matrice linkage, k optimal estimé par le plus grand saut).
    """
    linked = linkage(data, method=method)
    distances = linked[:, 2]
    sauts = np.diff(distances)
    idx_max = np.argmax(sauts)

    optimal_k = len(data) - (idx_max + 1)
    optimal_k = max(2, min(optimal_k, 6))
    h_cut = (distances[idx_max] + distances[idx_max + 1]) / 2

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(14, 6))
    dendrogram(linked, labels=labels, color_threshold=h_cut,
               leaf_rotation=90, leaf_font_size=8)
    plt.axhline(y=h_cut, color="r", linestyle="--",
                label=f"Coupure optimale (K={optimal_k})")
    plt.title(f"Dendrogramme — {title}")
    plt.legend()
    plt.tight_layout()
    safe = title.replace(" ", "_").replace("—", "").strip("_")
    plt.savefig(os.path.join(output_dir, f"dendrogram_{safe}.png"), dpi=100)
    plt.show()
    return linked, optimal_k


def do_hierarchical_clustering(
    data: np.ndarray,
    labels: list[str],
    method: str = "ward",
    title: str = "",
    output_dir: str = "outputs/clustering",
) -> pd.DataFrame:
    """
    Applique le clustering hiérarchique agglomératif et affiche le dendrogramme.

    Args:
        data:       Données standardisées.
        labels:     Noms des entreprises.
        method:     Méthode de liaison.
        title:      Titre des graphiques.
        output_dir: Dossier de sauvegarde.

    Returns:
        DataFrame avec colonnes de features + colonne 'cluster'.
    """
    print(f"\n=== Hierarchical Clustering — {title} ===")
    _, optimal_k = plot_dendrogram(data, labels, method, title, output_dir)

    agg = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
    cluster_labels = agg.fit_predict(data)

    result_df = pd.DataFrame(data, index=labels)
    result_df["cluster"] = cluster_labels
    print(result_df["cluster"].value_counts().sort_index())
    return result_df


# ─── Clustering sur les corrélations de rendements ───────────────────────────

def prepare_daily_returns_df(
    directory: str = DIR_HISTORICAL,
) -> pd.DataFrame:
    """
    Construit la matrice de corrélation des rendements journaliers.

    Args:
        directory: Dossier contenant les CSV historiques.

    Returns:
        Matrice de corrélation (entreprises × entreprises).
    """
    returns_dict: dict[str, pd.Series] = {}
    for path in glob.glob(os.path.join(directory, "*.csv")):
        name = os.path.basename(path).replace(".csv", "").replace("_", " ")
        try:
            df = pd.read_csv(path, index_col=0)
            if "Daily Return" in df.columns:
                returns_dict[name] = df["Daily Return"]
        except Exception as exc:
            print(f"  ✘ {name} : {exc}")

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.fillna(returns_df.mean())
    corr = returns_df.corr()
    distances = 1 - corr
    condensed = squareform(distances.values, checks=False)
    corr_matrix  = distances.values  
    labels = corr.columns.tolist()
    print(f"Matrice de corrélation : {corr.shape[0]} entreprises")
    return condensed, corr_matrix, labels


def cluster_by_returns_correlation(
    condensed: pd.DataFrame,
    labels: list[str],
    output_dir: str = "outputs/clustering",
) -> pd.Series:
    """
    Clustering hiérarchique sur la matrice de corrélation des rendements.

    Args:
        condensed: Matrice de distances (DataFrame carré).
        labels:             Noms des entreprises.
        output_dir:         Dossier de sauvegarde.

    Returns:
        Série {entreprise: cluster_id}.
    """
    print("\n=== Hierarchical Clustering — Rendements quotidiens ===")
    linked, optimal_k = plot_dendrogram(
        condensed, labels, "ward", "Rendements quotidiens", output_dir
    )
    cluster_assignments = fcluster(linked, t=optimal_k, criterion="maxclust")
    return pd.Series(cluster_assignments, index=labels, name="cluster")


# ─── DBSCAN ───────────────────────────────────────────────────────────────────

def do_dbscan_clustering(
    data: np.ndarray,
    labels: list[str],
    eps: float = 0.5,
    min_samples: int = 2,
    title: str = "",
    output_dir: str = "outputs/clustering",
) -> np.ndarray:
    """
    Applique DBSCAN avec recherche adaptative de l'epsilon optimal.

    Args:
        data:        Données standardisées.
        labels:      Noms des entreprises.
        eps:         Epsilon initial.
        min_samples: Nombre minimum de voisins.
        title:       Titre des graphiques.
        output_dir:  Dossier de sauvegarde.

    Returns:
        Tableau des labels de clusters (-1 = bruit).
    """
    print(f"\n=== DBSCAN (eps={eps}, min_samples={min_samples}) — {title} ===")

    is_correlation = "Corr" in title or "Rendements" in title
    grille_eps = [0.5, 0.4, 0.35, 0.3, 0.25, 0.55, 0.6] if is_correlation \
        else [1.5, 1.2, 1.0, 0.8, 1.8, 2.0, 2.2]

    best_eps, best_labels, n_clusters = eps, None, 0
    for test_eps in [eps] + grille_eps:
        lbls = DBSCAN(eps=test_eps, min_samples=min_samples).fit_predict(data)
        unique = set(lbls)
        n = len(unique) - (1 if -1 in unique else 0)
        if n >= 2:
            best_eps, best_labels, n_clusters = test_eps, lbls, n
            break

    if best_labels is None:
        best_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
        n_clusters = len(set(best_labels) - {-1})

    # Visualisation t-SNE
    perp = min(30, max(2, len(data) - 1))
    tsne_coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(data)

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab10", len(set(best_labels)))
    for i, lbl in enumerate(sorted(set(best_labels))):
        mask = best_labels == lbl
        if lbl == -1:
            plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                        c="black", marker="x", s=90, label="Bruit", alpha=0.7)
        else:
            plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                        color=colors(i), s=80, label=f"Cluster {lbl}", alpha=0.9)
    for idx, name in enumerate(labels):
        if idx < len(tsne_coords):
            plt.annotate(name, (tsne_coords[idx, 0], tsne_coords[idx, 1]),
                         fontsize=8, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    plt.title(f"t-SNE DBSCAN {title}\nEps={best_eps:.2f} | Clusters={n_clusters}")
    plt.legend(loc="best")
    plt.tight_layout()
    safe = title.replace(" ", "_").replace("—", "").strip("_")
    plt.savefig(os.path.join(output_dir, f"dbscan_tsne_{safe}.png"), dpi=100)
    plt.show()
    return best_labels


# ─── Comparaison des algorithmes ─────────────────────────────────────────────

def compare_clustering_algorithms(
    financial_ratios_df: pd.DataFrame,
    output_dir: str = "outputs/clustering",
) -> tuple[pd.DataFrame, dict]:
    """
    Compare K-Means, Hierarchical et DBSCAN sur trois types de données.

    Args:
        financial_ratios_df: DataFrame des ratios financiers.
        correlation_matrix:  Matrice de corrélation des rendements.
        output_dir:          Dossier de sauvegarde.

    Returns:
        (tableau des silhouette scores, dictionnaire des résultats bruts).
    """
    fin_scaled, fin_labels  = preprocess_for_clustering(financial_ratios_df, FINANCIAL_FEATURES)
    risk_scaled, risk_labels = preprocess_for_clustering(financial_ratios_df, RISK_FEATURES)
    condensed, corr_matrix, corr_labels = prepare_daily_returns_df()

    datasets = {
        "Financial Profiles":        (fin_scaled,  fin_labels),
        "Risk Profiles":             (risk_scaled, risk_labels),
        "Daily Returns Correlation": (corr_matrix, corr_labels),
    }

    perf: dict[str, dict] = {name: {} for name in datasets}

    for name, (data, labels) in datasets.items():
        print(f"\n{'='*60}\n{name}\n{'='*60}")

        # K-Means
        k = elbow_method(data, title=name, output_dir=output_dir)
        km_labels = do_kmeans_clustering(data, labels, k, title=name,
                                          output_dir=output_dir)["cluster"].values
        if len(set(km_labels)) > 1:
            perf[name]["K-Means"] = round(silhouette_score(data, km_labels), 4)

        # Hierarchical
        if name == "Daily Returns Correlation":
            hier_labels = cluster_by_returns_correlation(
                condensed, corr_labels).values
        else:
            hier_labels = do_hierarchical_clustering(
                data, labels, title=name, output_dir=output_dir)["cluster"].values
        if len(set(hier_labels)) > 1:
            perf[name]["Hierarchical"] = round(silhouette_score(data, hier_labels), 4)

        # DBSCAN
        db_labels = do_dbscan_clustering(data, labels, title=name, output_dir=output_dir)
        valid = db_labels != -1
        if len(set(db_labels[valid])) > 1 and valid.sum() > 1:
            perf[name]["DBSCAN"] = round(
                silhouette_score(data[valid] if hasattr(data, "__getitem__") else data, db_labels[valid]), 4
            )
        else:
            perf[name]["DBSCAN"] = float("nan")

    summary_df = pd.DataFrame(perf).T
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, "clustering_silhouette_scores.csv"))
    print("\n=== Silhouette Scores ===")
    print(summary_df)
    return summary_df, perf
