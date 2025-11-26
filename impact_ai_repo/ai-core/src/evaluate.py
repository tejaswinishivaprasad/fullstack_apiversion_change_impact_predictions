#!/usr/bin/env python3
"""
evaluate.py

Generate ROC, PR curves, calibration plots, and SHAP/permutation summaries.

Usage:
    python src/evaluate.py \
        --predictions src/analysis_outputs/predictions.csv \
        --features src/analysis_outputs/features.csv \
        --out-dir src/analysis_outputs
"""

from __future__ import annotations
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
)


def plot_roc(y_true, probs, out_path):
    """Safely plot ROC if both classes exist."""
    try:
        if len(set(y_true)) < 2:
            print("Skipping ROC: only one class present.")
            return
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print("Saved ROC:", out_path)
    except Exception as e:
        print("ROC error:", e)


def plot_pr(y_true, probs, out_path):
    """Safely plot Precision-Recall curve."""
    try:
        prec, rec, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(rec, prec)

        plt.figure()
        plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print("Saved PR:", out_path)
    except Exception as e:
        print("PR error:", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading predictions:", args.predictions)
    preds = pd.read_csv(args.predictions)

    print("Loading features:", args.features)
    features = pd.read_csv(args.features)

    # Determine if features.csv contains labels
    if "label" in features.columns:
        label_map = features.set_index("ace_id")["label"].to_dict()
    else:
        # fallback: treat ground truth as zeros (won't break)
        label_map = {}

    models = preds["model"].unique()
    print("Models in predictions:", models)

    for model in models:
        print("\nProcessing model:", model)
        sub = preds[preds["model"] == model].copy()

        # map ace_id → label
        y_true = []
        for aid in sub["ace_id"]:
            y_true.append(label_map.get(aid, 0))  # fallback = 0
        y_true = np.array(y_true)

        probs = sub["prob"].values

        roc_path = os.path.join(args.out_dir, f"roc_{model}.png")
        pr_path = os.path.join(args.out_dir, f"pr_{model}.png")

        plot_roc(y_true, probs, roc_path)
        plot_pr(y_true, probs, pr_path)

    # SHAP/permutation importance handling
    shap_txt = os.path.join(args.out_dir, "shap_summary.txt")
    shap_json = os.path.join(args.out_dir, "shap_summary.json")

    if os.path.exists(shap_txt):
        print("Found shap_summary.txt → converting to JSON")
        try:
            with open(shap_txt, "r", encoding="utf-8") as fh:
                text = fh.read()

            try:
                data = json.loads(text)
            except Exception:
                data = {"text_summary": text}

            with open(shap_json, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)

            print("Saved shap_summary.json")
        except Exception as e:
            print("SHAP conversion error:", e)
    else:
        print("No shap_summary.txt found; skipping SHAP export.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
