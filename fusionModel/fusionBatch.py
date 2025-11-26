"""
fusion_batch.py
----------------
Batch-level fusion of STATIC (image-based) and PROCESS (behavior-based)
malware detection model outputs.

This script:
 - Loads static predictions CSV
 - Loads process predictions CSV
 - Computes static malicious probability
 - Performs weighted or meta-classifier fusion
 - (Optional) trains meta-classifier
 - Produces fused output CSV
 - Evaluates performance if ground truth provided

Dependencies:
    pip install numpy pandas scikit-learn joblib
"""

import json
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import joblib
import os


# -------------------------------------------------------------
# COMPUTE STATIC MALICIOUS PROBABILITY
# -------------------------------------------------------------
def compute_static_mal_prob(probs, class_names):
    """
    Convert class probabilities → 1 malware score.

    If 'benign' exists → use 1 - P(benign)
    Else → fallback to max(probs)
    """
    probs = np.array(probs, dtype=float)

    benign_idx = None
    for i, c in enumerate(class_names):
        if str(c).lower() in ["benign", "clean", "normal", "good"]:
            benign_idx = i
            break

    if benign_idx is not None:
        return float(1.0 - probs[benign_idx])

    return float(probs.max())


# -------------------------------------------------------------
# EVALUATION METRICS
# -------------------------------------------------------------
def evaluate(true_labels, predicted_probs, threshold=0.5):
    predicted_labels = (predicted_probs >= threshold).astype(int)

    acc = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary"
    )
    try:
        auc = roc_auc_score(true_labels, predicted_probs)
    except Exception:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# -------------------------------------------------------------
# FUSION (Weighted or Meta)
# -------------------------------------------------------------
def weighted_fusion(static_p, process_p, w_s, w_p):
    if static_p is None:
        return process_p
    if process_p is None:
        return static_p
    return float(w_s * static_p + w_p * process_p)


def meta_fusion(model, static_p, process_p):
    X = np.array([[static_p, process_p]])
    return float(model.predict_proba(X)[0][1])


# -------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--static_csv", required=True,
                        help="CSV from static model (must include file_id)")
    parser.add_argument("--process_csv", required=True,
                        help="CSV from process model (must include file_id)")

    parser.add_argument("--classes", type=str, required=True,
                        help="JSON list of class names from static model")

    parser.add_argument("--out_csv", default="fusion_output.csv",
                        help="Output fused CSV")

    parser.add_argument("--weights", nargs=2, type=float, default=[0.6, 0.4],
                        help="Fusion weights: w_static w_process")

    parser.add_argument("--meta_model", type=str,
                        help="Path to pre-trained meta-classifier")

    parser.add_argument("--train_meta", type=str,
                        help="Path to save meta-classifier (train new LR)")

    parser.add_argument("--truth_csv", type=str,
                        help="Ground truth CSV: file_id,label")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load CSVs
    # ---------------------------------------------------------
    df_static = pd.read_csv(args.static_csv)
    df_process = pd.read_csv(args.process_csv)
    class_names = json.loads(args.classes)

    # ---------------------------------------------------------
    # Detect static probability columns
    # ---------------------------------------------------------
    # Option 1: many class columns: class_<name>
    class_cols = [c for c in df_static.columns if c.startswith("class_")]

    # Option 2: single JSON list column: "probs"
    probs_col = "probs" if "probs" in df_static.columns else None

    # ---------------------------------------------------------
    # Compute static malicious probability
    # ---------------------------------------------------------
    def extract_static_prob(row):
        if class_cols:
            probs = row[class_cols].values.astype(float)
            return compute_static_mal_prob(probs, class_names)

        if probs_col:
            probs = json.loads(row[probs_col])
            return compute_static_mal_prob(probs, class_names)

        return None

    df_static["static_mal_prob"] = df_static.apply(extract_static_prob, axis=1)

    # ---------------------------------------------------------
    # Merge static + process results by file_id
    # ---------------------------------------------------------
    df_merged = pd.merge(
        df_static,
        df_process[["file_id", "process_prob"]],
        on="file_id",
        how="left"
    )

    # ---------------------------------------------------------
    # FUSION
    # ---------------------------------------------------------
    if args.meta_model:
        model = joblib.load(args.meta_model)
        df_merged["fusion_prob"] = df_merged.apply(
            lambda r: meta_fusion(
                model,
                r["static_mal_prob"],
                r["process_prob"]
            ),
            axis=1,
        )
        fusion_method = "meta"
    else:
        w_static, w_process = args.weights
        df_merged["fusion_prob"] = df_merged.apply(
            lambda r: weighted_fusion(
                r["static_mal_prob"],
                r["process_prob"],
                w_static,
                w_process,
            ),
            axis=1,
        )
        fusion_method = "weighted"

    # ---------------------------------------------------------
    # SAVE FUSION CSV
    # ---------------------------------------------------------
    df_merged.to_csv(args.out_csv, index=False)
    print(f"[✔] Fused results saved to: {args.out_csv}")
    print(f"[✔] Fusion method used: {fusion_method}")

    # ---------------------------------------------------------
    # EVALUATION (if labels available)
    # ---------------------------------------------------------
    if args.truth_csv:
        df_truth = pd.read_csv(args.truth_csv)
        df_eval = pd.merge(
            df_merged,
            df_truth[["file_id", "label"]],
            on="file_id",
            how="inner"
        )

        metrics = evaluate(
            df_eval["label"].values.astype(int),
            df_eval["fusion_prob"].values
        )

        print("\n=== Evaluation Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # -----------------------------------------------
        # TRAIN META CLASSIFIER
        # -----------------------------------------------
        if args.train_meta:
            X = df_eval[["static_mal_prob", "process_prob"]].values
            y = df_eval["label"].astype(int).values

            clf = LogisticRegression(max_iter=2000)
            clf.fit(X, y)

            joblib.dump(clf, args.train_meta)
            print(f"\n[✔] Meta-classifier trained and saved to: {args.train_meta}")


if __name__ == "__main__":
    main()
