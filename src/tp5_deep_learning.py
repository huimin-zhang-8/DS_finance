"""
TP5 — Réseaux de neurones (MLP, RNN, LSTM) pour la prédiction J+1.

Fonctions exportées :
    build_mlp_model, build_rnn_model, build_lstm_model,
    train_model, predict, run_deep_learning_comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout

from configs.settings import N_DAYS


# ─── Constructeurs de modèles ─────────────────────────────────────────────────

def _get_optimizer(optimizer, lr: float):
    """Résout une chaîne ou un objet optimiseur et applique le learning rate."""
    if isinstance(optimizer, str):
        opt = tf.keras.optimizers.get(optimizer)
    else:
        opt = optimizer
    opt.learning_rate = lr
    return opt


def build_mlp_model(
    input_dim: int,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.2,
    activation: str = "relu",
    optimizer: str = "adam",
    lr: float = 1e-3,
) -> Sequential:
    """
    Construit un réseau de neurones dense (MLP) pour la régression.

    Args:
        input_dim:    Dimension de l'entrée (= N_DAYS features).
        hidden_dims:  Nombre de neurones par couche cachée.
        dropout_rate: Taux de dropout entre les couches.
        activation:   Fonction d'activation des couches cachées.
        optimizer:    Nom ou instance de l'optimiseur.
        lr:           Taux d'apprentissage.

    Returns:
        Modèle Keras compilé.
    """
    model = Sequential()
    model.add(Dense(hidden_dims[0], activation=activation, input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    for h in hidden_dims[1:]:
        model.add(Dense(h, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=_get_optimizer(optimizer, lr), loss="mean_squared_error")
    return model


def build_rnn_model(
    input_shape: tuple[int, int],
    hidden_dims: tuple[int, ...] = (64,),
    dropout_rate: float = 0.2,
    activation: str = "tanh",
    optimizer: str = "adam",
    lr: float = 1e-3,
) -> Sequential:
    """
    Construit un réseau récurrent simple (SimpleRNN).

    Args:
        input_shape:  (time_steps, n_features).
        hidden_dims:  Nombre de cellules par couche RNN.
        dropout_rate: Taux de dropout.
        activation:   Fonction d'activation.
        optimizer:    Optimiseur.
        lr:           Taux d'apprentissage.

    Returns:
        Modèle Keras compilé.
    """
    model = Sequential()
    for i, h in enumerate(hidden_dims):
        return_seq = i < len(hidden_dims) - 1
        kwargs = {"return_sequences": return_seq, "activation": activation}
        if i == 0:
            kwargs["input_shape"] = input_shape
        model.add(SimpleRNN(h, **kwargs))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=_get_optimizer(optimizer, lr), loss="mean_squared_error")
    return model


def build_lstm_model(
    input_shape: tuple[int, int],
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.2,
    activation: str = "tanh",
    optimizer: str = "adam",
    lr: float = 1e-3,
) -> Sequential:
    """
    Construit un réseau LSTM.

    Args:
        input_shape:  (time_steps, n_features).
        hidden_dims:  Nombre de cellules par couche LSTM.
        dropout_rate: Taux de dropout.
        activation:   Fonction d'activation.
        optimizer:    Optimiseur.
        lr:           Taux d'apprentissage.

    Returns:
        Modèle Keras compilé.
    """
    model = Sequential()
    for i, h in enumerate(hidden_dims):
        return_seq = i < len(hidden_dims) - 1
        kwargs = {"return_sequences": return_seq, "activation": activation}
        if i == 0:
            kwargs["input_shape"] = input_shape
        model.add(LSTM(h, **kwargs))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=_get_optimizer(optimizer, lr), loss="mean_squared_error")
    return model


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_shape,
    epochs: int = 20,
    batch_size: int = 32,
    **kwargs,
) -> Sequential:
    """
    Construit et entraîne un modèle MLP, RNN ou LSTM.

    Args:
        model_type:  'MLP', 'RNN' ou 'LSTM'.
        X_train:     Données d'entraînement.
        y_train:     Labels d'entraînement.
        input_shape: int pour MLP, tuple (steps, features) pour RNN/LSTM.
        epochs:      Nombre d'époques.
        batch_size:  Taille des batchs.
        **kwargs:    Hyperparamètres passés aux builders (hidden_dims, dropout_rate, …).

    Returns:
        Modèle Keras entraîné.
    """
    if model_type == "MLP":
        dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        print(f"--- MLP (input_dim={dim}) ---")
        model = build_mlp_model(input_dim=dim, **kwargs)
    elif model_type == "RNN":
        print(f"--- RNN (input_shape={input_shape}) ---")
        model = build_rnn_model(input_shape=input_shape, **kwargs)
    elif model_type == "LSTM":
        print(f"--- LSTM (input_shape={input_shape}) ---")
        model = build_lstm_model(input_shape=input_shape, **kwargs)
    else:
        raise ValueError(f"model_type inconnu : {model_type!r} (MLP | RNN | LSTM)")

    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    print("Entraînement terminé.\n")
    return model


# ─── Évaluation ───────────────────────────────────────────────────────────────

def predict(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    model_type: str,
) -> dict:
    """
    Applique le modèle, inverse la normalisation et calcule MAE / RMSE.

    Args:
        model:      Modèle Keras entraîné.
        X_test:     Features de test.
        y_test:     Cibles de test (normalisées).
        scaler:     MinMaxScaler fitté sur le train.
        model_type: Nom du modèle (pour l'affichage).

    Returns:
        Dictionnaire {model_type, MAE, RMSE, y_pred, y_true}.
    """
    print(f"=== Évaluation — {model_type} ===")
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")

    print(f"\n{'Index':<8} | {'Réel':<12} | {'Prédit':<12} | {'Écart':<10}")
    print("-" * 48)
    for i in range(min(10, len(y_true))):
        print(f"{i:<8} | {y_true[i]:<12.2f} | {y_pred[i]:<12.2f} | {abs(y_true[i]-y_pred[i]):<10.2f}")

    return {"model_type": model_type, "MAE": round(mae, 2),
            "RMSE": round(rmse, 2), "y_pred": y_pred, "y_true": y_true}


# ─── Comparaison des architectures ───────────────────────────────────────────

ARCHITECTURES: dict[str, list[dict]] = {
    "MLP":  [{"hidden_dims": (32,),      "dropout_rate": 0.0},
             {"hidden_dims": (128, 64),  "dropout_rate": 0.2}],
    "RNN":  [{"hidden_dims": (32,),      "dropout_rate": 0.0},
             {"hidden_dims": (64, 32),   "dropout_rate": 0.1}],
    "LSTM": [{"hidden_dims": (50,),      "dropout_rate": 0.0},
             {"hidden_dims": (100, 50),  "dropout_rate": 0.2}],
}


def run_deep_learning_comparison(
    reg_data: dict,
    output_dir: str = "outputs/regression",
    epochs: int = 15,
    batch_size: int = 32,
) -> tuple[dict, pd.DataFrame]:
    """
    Compare plusieurs architectures pour chaque type de modèle (MLP, RNN, LSTM)
    et retourne le meilleur modèle de chaque type.

    Args:
        reg_data:   Données de régression (retour de prepare_regression_data).
        output_dir: Dossier de sauvegarde.
        epochs:     Nombre d'époques par essai.
        batch_size: Taille des batchs.

    Returns:
        (best_models dict, results_df DataFrame avec MAE/MSE/RMSE).
    """
    os.makedirs(output_dir, exist_ok=True)
    scaler  = reg_data["scaler"]
    X_train = reg_data["X_train"]
    y_train = reg_data["y_train"]
    X_test  = reg_data["X_test"]
    y_test  = reg_data["y_test"]

    # Tenseurs 3D pour RNN/LSTM
    X_train_3d = np.expand_dims(X_train, axis=-1)
    X_test_3d  = np.expand_dims(X_test,  axis=-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    best_models: dict[str, Sequential] = {}
    dl_predictions: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    for m_type in ("MLP", "RNN", "LSTM"):
        best_rmse, best_model_obj = float("inf"), None
        is_seq = m_type in ("RNN", "LSTM")
        in_shape = (X_train_3d.shape[1], 1) if is_seq else (X_train.shape[1],)
        X_tr = X_train_3d if is_seq else X_train
        X_te = X_test_3d  if is_seq else X_test

        print(f"\n--- Recherche meilleure architecture {m_type} ---")
        for config in ARCHITECTURES[m_type]:
            model = train_model(
                model_type=m_type, X_train=X_tr, y_train=y_train,
                input_shape=in_shape,
                activation="relu" if m_type == "MLP" else "tanh",
                optimizer="adam", lr=1e-3,
                epochs=epochs, batch_size=batch_size,
                **config,
            )
            preds = model.predict(X_te, verbose=0).flatten()
            rmse_scaled = np.sqrt(mean_squared_error(y_test, preds))
            if rmse_scaled < best_rmse:
                best_rmse, best_model_obj = rmse_scaled, model

        best_models[m_type] = best_model_obj
        y_pred_scaled = best_model_obj.predict(X_te, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        dl_predictions[m_type] = y_pred

        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        rows.append({"Modèle": m_type, "MAE": round(mae, 2),
                     "MSE": round(mse, 2), "RMSE": round(rmse, 2)})
        print(f"\n{m_type} — MAE={mae:.2f}  RMSE={rmse:.2f}")

    # Graphique comparatif
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true, color="black", label="Valeurs réelles", linewidth=2)
    for (m_type, y_pred), col in zip(dl_predictions.items(), ["blue", "orange", "green"]):
        ax.plot(y_pred, color=col, label=f"Prédiction {m_type}", alpha=0.7)
    ax.set_title("Prédictions J+1 — MLP vs RNN vs LSTM")
    ax.set_xlabel("Jours")
    ax.set_ylabel("Prix ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "deep_learning_comparison.png"), dpi=100)
    plt.show()

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "deep_learning_performance.csv"), index=False)
    return best_models, results_df
