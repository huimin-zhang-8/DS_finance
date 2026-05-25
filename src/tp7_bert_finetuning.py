"""
TP7 — Fine-tuning BERT / FinBERT pour l'analyse de sentiment financier.

Datasets HuggingFace utilisés :
    - zeroshot/twitter-financial-news-sentiment
    - nickmuchi/financial-classification

Fonctions exportées :
    load_and_merge_datasets, finetune_model, compare_bert_results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from configs.settings import BERT_MODEL_NAME, FINBERT_MODEL_NAME, DIR_MODELS


# ─── Chargement et fusion des datasets ────────────────────────────────────────

def load_and_merge_datasets() -> DatasetDict:
    """
    Charge et fusionne les deux datasets de sentiment financier HuggingFace.

    Datasets :
        - zeroshot/twitter-financial-news-sentiment  (tweets financiers)
        - nickmuchi/financial-classification          (phrases économiques)

    Les deux partagent le schéma {text, label} avec 3 classes :
        0 = Négatif, 1 = Neutre, 2 = Positif

    Returns:
        DatasetDict avec clés 'train' et 'test'.
    """
    ds1 = load_dataset("zeroshot/twitter-financial-news-sentiment")
    ds2 = load_dataset("nickmuchi/financial-classification")

    ds1_train = ds1["train"].select_columns(["text", "label"])
    ds1_test  = ds1["validation"].select_columns(["text", "label"])

    ds2_train = ds2["train"].rename_column("labels", "label").select_columns(["text", "label"])
    ds2_test  = ds2["test"].rename_column("labels", "label").select_columns(["text", "label"])

    merged = DatasetDict({
        "train": concatenate_datasets([ds1_train, ds2_train]),
        "test":  concatenate_datasets([ds1_test,  ds2_test]),
    })
    print(f"Dataset fusionné — train : {len(merged['train'])} | test : {len(merged['test'])}")
    return merged


# ─── Fine-tuning ──────────────────────────────────────────────────────────────

def finetune_model(
    model_name: str,
    dataset: DatasetDict,
    batch_size: int = 16,
    num_epochs: int = 3,
    output_dir: str = DIR_MODELS,
) -> tuple[dict, str]:
    """
    Fine-tune un modèle BERT ou FinBERT sur le dataset de sentiment financier.

    Args:
        model_name:  Identifiant HuggingFace (ex: 'bert-base-uncased').
        dataset:     DatasetDict issu de load_and_merge_datasets.
        batch_size:  Taille des batchs d'entraînement et d'évaluation.
        num_epochs:  Nombre d'époques d'entraînement.
        output_dir:  Répertoire racine de sauvegarde des checkpoints et du modèle final.

    Returns:
        (metrics dict, chemin du modèle sauvegardé).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_train = dataset["train"].map(tokenize, batched=True)
    tokenized_test  = dataset["test"].map(tokenize,  batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

    model_short = model_name.replace("/", "_")
    checkpoints_dir = os.path.join(output_dir, f"{model_short}_results")

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        report_to="none",
        disable_tqdm=False,
        logging_first_step=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    print(f"\n=== Fine-tuning : {model_name} ===")
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    save_path = os.path.join(output_dir, f"best_model_{model_short}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  → Modèle sauvegardé : {save_path}")
    return metrics, save_path


# ─── Comparaison BERT vs FinBERT ─────────────────────────────────────────────

def compare_bert_results(
    bert_metrics: dict,
    finbert_metrics: dict,
    output_dir: str = "outputs/sentiment",
) -> pd.DataFrame:
    """
    Compare et visualise les performances de BERT et FinBERT.

    Args:
        bert_metrics:    Dictionnaire de métriques retourné par finetune_model pour BERT.
        finbert_metrics: Idem pour FinBERT.
        output_dir:      Dossier de sauvegarde.

    Returns:
        DataFrame des métriques comparatives.
    """
    data = {
        "Metric":  ["Accuracy", "F1-Score", "Precision", "Recall"],
        "BERT":    [bert_metrics.get(f"eval_{k}", float("nan"))
                    for k in ("accuracy", "f1", "precision", "recall")],
        "FinBERT": [finbert_metrics.get(f"eval_{k}", float("nan"))
                    for k in ("accuracy", "f1", "precision", "recall")],
    }
    df = pd.DataFrame(data).set_index("Metric")
    print("\n=== BERT vs FinBERT ===")
    print(df.round(4))

    os.makedirs(output_dir, exist_ok=True)
    ax = df.plot(kind="bar", figsize=(10, 6), rot=0, color=["#AEC6CF", "#FFB347"])
    plt.title("Comparaison des performances : BERT vs FinBERT", fontsize=14)
    plt.ylabel("Score (0 à 1)")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    for p in ax.patches:
        ax.annotate(
            str(round(p.get_height(), 3)),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="center", xytext=(0, 10), textcoords="offset points",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bert_vs_finbert.png"), dpi=100)
    plt.show()

    df.to_csv(os.path.join(output_dir, "bert_comparison.csv"))
    return df
