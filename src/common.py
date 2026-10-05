"""
IT2011 Machine Learning Project: Shared Utilities Module
Group ID: 2026-Y2-S1-MLB-B1G1-07
Dataset: Adult Income Dataset (UCI ML Repository)

Provides shared, standardized functions across all group members:
- Consistent data loading
- Leakage-free Stratified 80/20 train/test split (random_state=42)
- Comprehensive evaluation metrics calculation
- Publication-quality plotting utilities (Confusion Matrix, ROC, PR curves)
- Model result serialization to results/model_results/<IT_NUMBER>.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "outputs", "adult_processed.csv")

def load_data(filepath=None):
    """
    Loads the processed adult dataset.
    Returns:
        X (pd.DataFrame): Input features
        y (pd.Series): Target variable (0: <=50K, 1: >50K)
    """
    path = filepath or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        # Fallback check from current working directory
        fallbacks = [
            "results/outputs/adult_processed.csv",
            "../results/outputs/adult_processed.csv",
            "../../results/outputs/adult_processed.csv"
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                path = fb
                break
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at '{path}'. Ensure group_pipeline.ipynb has run.")
        
    df = pd.read_csv(path)
    X = df.drop(columns=['income'])
    y = df['income'].astype(int)
    return X, y

def get_train_test_split(X, y, test_size=0.20, random_state=42):
    """
    Standardized 80/20 Stratified train-test split enforced across all group members.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Computes all standard classification metrics required by the SLIIT evaluation rubric.
    """
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Renders and optionally saves a normalized confusion matrix plot.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(False)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_roc_pr_curves(y_true, y_proba, model_name="Model", save_path=None):
    """
    Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves side by side.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    ax_roc.plot(fpr, tpr, color='#2980b9', lw=2, label=f'{model_name} (AUC = {auc_score:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, linestyle='--', alpha=0.5)

    # PR
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
    ax_pr.plot(rec_curve, prec_curve, color='#27ae60', lw=2, label='PR Curve')
    ax_pr.set_title('Precision-Recall Curve (>50K Class)', fontweight='bold')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(loc='upper right')
    ax_pr.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def save_member_result(it_number, model_name, best_hyperparameters, metrics, save_dir="results/model_results"):
    """
    Serializes individual optimum model result into a standardized JSON file for group comparison.
    """
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "it_number": it_number,
        "model_name": model_name,
        "best_hyperparameters": best_hyperparameters,
        "metrics": metrics
    }
    file_path = os.path.join(save_dir, f"{it_number}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"[+] Saved model result for {it_number} ({model_name}) to: {file_path}")
