# Projet Pratique de la Data Science

**Groupe : Huimin ZHANG, Chloé DJOUSSA, Astrid BOUA**

Pipeline d'analyse quotidienne des marchés financiers combinant clustering, classification, régression et analyse de sentiment pour générer des recommandations d'investissement.

---

## Structure du projet

```
mon_projet/
│
├── main.py                          
├── requirements.txt                 
├── README.md                        
│
├── configs/
│   ├── __init__.py
│   └── settings.py
│
├── src/
│   ├── __init__.py
│   ├── tp1_scraping.py
│   ├── tp2_clustering.py
│   ├── tp3_classification.py
│   ├── tp4_regression.py
│   ├── tp5_deep_learning.py
│   ├── tp6_news_scraping.py
│   ├── tp7_bert_finetuning.py
│   └── tp8_sentiment_analysis.py
│
├── data/                            ← généré automatiquement
│   ├── raw/
│   │   ├── companies_historical_data/
│   │   └── news_data/
│   └── processed/
│
├── models/                          ← généré automatiquement
│
├── outputs/                         ← généré automatiquement
│   ├── clustering/
│   ├── classification/
│   ├── regression/
│   ├── sentiment/
│   └── recommendations/
│
└── notebooks/
    └── projet.ipynb
```

---

## Contenu de chaque dossier

### `configs/`
Paramètres globaux partagés par tous les modules.

`settings.py` contient :
- La clé API NewsAPI
- La liste des 41 entreprises analysées et leurs tickers
- Les dates de début et de fin du scraping
- Les features utilisées pour le clustering
- Les seuils de recommandation BUY / SELL
- Les pondérations de la stratégie d'agrégation

### `src/`
Un fichier par TP. Chaque fichier est autonome et contient uniquement des fonctions réutilisables, sans code exécuté à l'import.

| Fichier | Rôle |
|---|---|
| `tp1_scraping.py` | Scraping des ratios financiers et de l'historique de prix via yfinance |
| `tp2_clustering.py` | Clustering K-Means, Hiérarchique et DBSCAN sur profils financiers, de risque et de rendements |
| `tp3_classification.py` | Classification Buy/Hold/Sell avec XGBoost, Random Forest, KNN, Régression Logistique, SVM |
| `tp4_regression.py` | Régression ML pour la prédiction du prix à J+1 |
| `tp5_deep_learning.py` | Réseaux de neurones MLP, RNN, LSTM pour la prédiction du prix à J+1 |
| `tp6_news_scraping.py` | Scraping des news financières via NewsAPI |
| `tp7_bert_finetuning.py` | Fine-tuning de BERT et FinBERT sur un dataset de sentiment financier |
| `tp8_sentiment_analysis.py` | Analyse de sentiment des news et visualisation vs cours boursiers |

### `data/`
Données brutes et traitées, générées automatiquement au premier lancement.

```
data/
├── raw/
│   ├── companies_historical_data/   ← un CSV par entreprise (5 ans de prix)
│   └── news_data/                   ← un JSON par entreprise (articles de presse)
└── processed/
    └── financial_ratios.csv         ← ratios financiers de toutes les entreprises
```

### `models/`
Modèles entraînés sauvegardés sur disque. Si les modèles existent déjà, le pipeline les recharge sans les ré-entraîner.

```
models/
├── best_model_bert-base-uncased/    ← BERT fine-tuné (généré par TP7)
└── best_model_ProsusAI_finbert/     ← FinBERT fine-tuné (généré par TP7)
```

Ce dossier n'a pas été push sur GitHub par soucis d'espace. 

### `outputs/`
Tous les graphiques et fichiers de résultats générés par la pipeline.

```
outputs/
├── clustering/          ← courbes d'inertie, t-SNE, dendrogrammes, silhouette scores
├── classification/      ← rapports de classification, SHAP plots
├── regression/          ← courbes prédictions vs réel, tableau de métriques
├── sentiment/           ← graphiques TP8, comparaison BERT vs FinBERT
└── recommendations/     ← recommendations_YYYY-MM-DD.csv, dashboard
```

### `notebooks/`
Le notebook Jupyter sur lequel on a travaillé tout au long du projet afin de visualiser direcement ce que nous faisions, avant de transformer le projet sous format .py. 

---

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/votre-repo/mon-projet.git
cd mon-projet

# 2. Créer l'environnement Conda
conda create -n mon_projet python=3.10
conda activate mon_projet

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Configuration

Avant de lancer le pipeline, renseigne la clé API NewsAPI dans `configs/settings.py` :

```python
NEWS_API_KEY = "cle_API"
```

---

## Lancement

### Premier lancement (tout from scratch)
```bash
python main.py
```
Scrape les données, entraîne tous les modèles et génère les recommandations.

### Lancement quotidien (données et modèles déjà présents)
```bash
python main.py --skip-scraping --skip-training
```
Recharge les données et modèles existants, scrape uniquement les nouvelles news, génère les recommandations du jour.

### Mode démo (sur N entreprises seulement)
```bash
python main.py --demo 5
```
Utile pour tester le pipeline rapidement sur les 5 premières entreprises.

---

## Ce que produit le pipeline

À chaque exécution, le pipeline génère dans `outputs/recommendations/` :

**`recommendations_YYYY-MM-DD.csv`** — une ligne par entreprise avec :

| Colonne | Description |
|---|---|
| `company` | Nom de l'entreprise |
| `symbol` | Symbole boursier |
| `classification` | Signal Buy/Hold/Sell du modèle de classification |
| `predicted_price` | Prix prédit à J+1 (en devise locale) |
| `predicted_return` | Rendement prédit à J+1 (en %) |
| `today_news_titles` | Liste des titres d'articles du jour (liste vide si aucune news) |
| `today_news_count` | Nombre d'articles du jour |
| `sentiment_score` | Score de sentiment des news (-1 à +1) |
| `similar_companies` | Liste des entreprises appartenant au même cluster |
| `final_score` | Score agrégé entre -1 et +1 |
| `recommendation` | Recommandation finale : BUY / HOLD / SELL |

**`dashboard_YYYY-MM-DD.png`** — graphique visuel des scores par entreprise.

---

## Stratégie d'agrégation des signaux

Le score final est une moyenne pondérée de trois signaux :

```
score = 0.40 × signal_classification
      + 0.35 × signal_régression
      + 0.25 × signal_sentiment
```

| Signal | Source | Poids |
|---|---|---|
| Classification | Modèle Buy/Hold/Sell (TP3) | 40% |
| Régression | Rendement prédit J+1 (TP4) | 35% |
| Sentiment | Analyse des news du jour (TP8) | 25% |

**Règles de décision :**
- `score > +0.15` → **BUY**
- `score < -0.15` → **SELL**
- sinon → **HOLD**

---

## Dépendances principales

| Librairie | Usage |
|---|---|
| `yfinance` | Scraping des prix et ratios financiers |
| `scikit-learn` | Clustering, classification, régression |
| `xgboost` | XGBoost classifier et regressor |
| `tensorflow` | Réseaux MLP, RNN, LSTM |
| `transformers` | BERT et FinBERT |
| `ta` | Indicateurs techniques (RSI, MACD, Bollinger…) |
| `shap` | Explicabilité des modèles |
| `newsapi` | Scraping des news financières |
