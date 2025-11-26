import json
import numpy as np
import pandas as pd
from typing import Dict, Optional

# Optional import for meta-classifier
try:
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# --------------------------------------------------------------
# python fusionLayer.py --resnet_csv path_to_resnet_predictions.csv \ --efficientnet_csv path_to_efficientnet_predictions.csv \ --process_prob 0.85 \ --weights 0.6 0.4
# --------------------------------------------------------------

# -------------------------------------------------------------
#  COMPUTE STATIC MALICIOUS PROBABILITY
# -------------------------------------------------------------
def compute_static_mal_prob(static_probs: list, class_names: list) -> float:
    """
    Convert class probabilities → a single malicious probability.
    If 'benign' exists, use 1 - P(benign).
    Else, fallback to max(probabilities).
    """
    static_probs = np.array(static_probs, dtype=float)

    benign_idx = None
    for i, name in enumerate(class_names):
        if name.lower() in ["benign", "clean", "normal", "good"]:
            benign_idx = i
            break

    if benign_idx is not None:
        return float(1.0 - static_probs[benign_idx])

    return float(static_probs.max())


# -------------------------------------------------------------
#  WEIGHTED FUSION
# -------------------------------------------------------------
def weighted_fusion(
    static_prob: Optional[float],
    process_prob: Optional[float],
    w_static: float = 0.6,
    w_process: float = 0.4
) -> float:
    """
    Combine static model and process model predictions using weighted average.
    """
    if static_prob is None and process_prob is None:
        return 0.0
    if static_prob is None:
        return process_prob
    if process_prob is None:
        return static_prob

    return float(w_static * static_prob + w_process * process_prob)


# -------------------------------------------------------------
#  META-CLASSIFIER FUSION
# -------------------------------------------------------------
def meta_fusion(meta_model_path: str, static_prob: float, process_prob: float) -> float:
    """
    Uses logistic regression or other ML classifier to fuse scores.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not installed. Cannot run meta fusion.")

    model = joblib.load(meta_model_path)
    X = np.array([[static_prob, process_prob]])
    return float(model.predict_proba(X)[0][1])  # P(malicious)


# -------------------------------------------------------------
#  DECISION LOGIC
# -------------------------------------------------------------
def make_decision(score: float, high=0.80, medium=0.50) -> str:
    """
    Decide the verdict based on the final score.
    """
    if score >= high:
        return "MALICIOUS"
    if score >= medium:
        return "SUSPICIOUS"
    return "BENIGN"


# -------------------------------------------------------------
#  FUSION LOGIC TO COMBINE STATIC PREDICTIONS (ENSMBLE)
# -------------------------------------------------------------
def fuse_predictions_from_csv(
    resnet_csv: str,
    efficientnet_csv: str,
    process_output: Dict = None,
    class_names: Optional[list] = None,
    weights: tuple = (0.6, 0.4),
    meta_model_path: Optional[str] = None
) -> Dict:
    """
    Load static model predictions from CSVs, combine with process model output, and return final decision.
    """
    # Load the ResNet50 and EfficientNet prediction CSVs
    resnet_preds = pd.read_csv(resnet_csv)
    efficientnet_preds = pd.read_csv(efficientnet_csv)

    # Ensure both models have the same files (based on the file_id or another common identifier)
    assert all(resnet_preds["file_id"] == efficientnet_preds["file_id"]), "File IDs don't match between models!"

    # Extract the predictions from each model (assumes the columns 'class_probs' and 'predicted_family')
    static_probs_resnet = resnet_preds["class_probs"].tolist()
    static_probs_efficientnet = efficientnet_preds["class_probs"].tolist()

    # Compute static malicious probabilities
    static_prob_resnet = compute_static_mal_prob(static_probs_resnet, class_names)
    static_prob_efficientnet = compute_static_mal_prob(static_probs_efficientnet, class_names)

    # ---------------------
    #  Extract process info
    # ---------------------
    process_prob = None
    if process_output:
        process_prob = process_output.get("malicious_prob")

    # ---------------------
    #  Fusion stage
    # ---------------------
    if meta_model_path:
        final_score = meta_fusion(
            meta_model_path,
            static_prob_resnet if static_prob_resnet else 0.0,
            static_prob_efficientnet if static_prob_efficientnet else 0.0
        )
        fusion_method = "meta_classifier"
    else:
        w_static, w_process = weights
        final_score = weighted_fusion(static_prob_resnet, static_prob_efficientnet, w_static, w_process)
        fusion_method = "weighted"

    # ---------------------
    #  Final decision
    # ---------------------
    verdict = make_decision(final_score)

    # ---------------------
    #  Output JSON
    # ---------------------
    return {
        "static": {
            "resnet": {
                "P_malicious": static_prob_resnet,
                "predicted_family": resnet_preds["predicted_family"].iloc[0],
            },
            "efficientnet": {
                "P_malicious": static_prob_efficientnet,
                "predicted_family": efficientnet_preds["predicted_family"].iloc[0],
            },
        },
        "process": {
            "P_malicious": process_prob,
        },
        "fusion": {
            "method": fusion_method,
            "weights": {"static": weights[0], "process": weights[1]} if fusion_method == "weighted" else None,
            "final_score": final_score,
            "verdict": verdict
        }
    }


# -------------------------------------------------------------
# CLI FOR STANDALONE TESTING
# -------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fusion Layer CLI")

    parser.add_argument("--resnet_csv", type=str, required=True, help="Path to ResNet50 predictions CSV")
    parser.add_argument("--efficientnet_csv", type=str, required=True, help="Path to EfficientNet predictions CSV")
    parser.add_argument("--process_prob", type=float, help="Process malicious probability")
    parser.add_argument("--meta_model_path", type=str, help="Path to meta-classifier")
    parser.add_argument("--weights", nargs=2, type=float, default=[0.6, 0.4])
    parser.add_argument("--class_names", type=str, help="JSON list of classes")
    args = parser.parse_args()

    # Parse the static model predictions CSV files
    result = fuse_predictions_from_csv(
        resnet_csv=args.resnet_csv,
        efficientnet_csv=args.efficientnet_csv,
        process_output={"malicious_prob": args.process_prob} if args.process_prob else None,
        class_names=json.loads(args.class_names) if args.class_names else None,
        weights=tuple(args.weights),
        meta_model_path=args.meta_model_path
    )

    print(json.dumps(result, indent=2))


# """
# fusion_layer.py
# ----------------
# Fusion layer for combining STATIC (image-based malware detection)
# and PROCESS (behavior-based malware detection) model predictions.
#
# Supports:
#  - Weighted fusion
#  - Optional meta-classifier (Logistic Regression)
#  - Fallback if one modality is missing
#  - Clean JSON output for UI / logs
# """
#
# import json
# import numpy as np
# from typing import Dict, Optional
#
# # Optional import for meta-classifier
# try:
#     import joblib
#     SKLEARN_AVAILABLE = True
# except ImportError:
#     SKLEARN_AVAILABLE = False
#
#
# # -------------------------------------------------------------
# #  COMPUTE STATIC MALICIOUS PROBABILITY
# # -------------------------------------------------------------
# def compute_static_mal_prob(static_probs: list, class_names: list) -> float:
#     """
#     Convert class probabilities → a single malicious probability.
#     If 'benign' exists, use 1 - P(benign).
#     Else, fallback to max(probabilities).
#     """
#     static_probs = np.array(static_probs, dtype=float)
#
#     benign_idx = None
#     for i, name in enumerate(class_names):
#         if name.lower() in ["benign", "clean", "normal", "good"]:
#             benign_idx = i
#             break
#
#     if benign_idx is not None:
#         return float(1.0 - static_probs[benign_idx])
#
#     return float(static_probs.max())
#
#
# # -------------------------------------------------------------
# #  WEIGHTED FUSION
# # -------------------------------------------------------------
# def weighted_fusion(
#     static_prob: Optional[float],
#     process_prob: Optional[float],
#     w_static: float = 0.6,
#     w_process: float = 0.4
# ) -> float:
#
#     if static_prob is None and process_prob is None:
#         return 0.0
#     if static_prob is None:
#         return process_prob
#     if process_prob is None:
#         return static_prob
#
#     return float(w_static * static_prob + w_process * process_prob)
#
#
# # -------------------------------------------------------------
# #  META-CLASSIFIER FUSION
# # -------------------------------------------------------------
# def meta_fusion(meta_model_path: str, static_prob: float, process_prob: float) -> float:
#     """
#     Uses logistic regression or other ML classifier to fuse scores.
#     """
#     if not SKLEARN_AVAILABLE:
#         raise RuntimeError("scikit-learn is not installed. Cannot run meta fusion.")
#
#     model = joblib.load(meta_model_path)
#     X = np.array([[static_prob, process_prob]])
#     return float(model.predict_proba(X)[0][1])  # P(malicious)
#
#
# # -------------------------------------------------------------
# #  DECISION LOGIC
# # -------------------------------------------------------------
# def make_decision(score: float, high=0.80, medium=0.50) -> str:
#     if score >= high:
#         return "MALICIOUS"
#     if score >= medium:
#         return "SUSPICIOUS"
#     return "BENIGN"
#
#
# # -------------------------------------------------------------
# #  MAIN FUSION FUNCTION
# # -------------------------------------------------------------
# def fuse_predictions(
#     static_output: Dict = None,
#     process_output: Dict = None,
#     class_names: Optional[list] = None,
#     weights: tuple = (0.6, 0.4),
#     meta_model_path: Optional[str] = None
# ) -> Dict:
#     """
#     static_output format:
#         {
#             "class_probs": [...],
#             "predicted_family": "Ramnit",
#             "confidence": 0.91
#         }
#
#     process_output format:
#         {
#             "malicious_prob": 0.82,
#             "pid": 1234
#         }
#     """
#     # ---------------------
#     #  Extract static info
#     # ---------------------
#     static_prob = None
#     pred_family = None
#     family_conf = None
#
#     if static_output:
#         probs = static_output.get("class_probs")
#         pred_family = static_output.get("predicted_family")
#         family_conf = static_output.get("confidence")
#
#         if probs is not None and class_names is not None:
#             static_prob = compute_static_mal_prob(probs, class_names)
#
#     # ---------------------
#     #  Extract process info
#     # ---------------------
#     process_prob = None
#     if process_output:
#         process_prob = process_output.get("malicious_prob")
#
#     # ---------------------
#     #  Fusion stage
#     # ---------------------
#     if meta_model_path:
#         final_score = meta_fusion(
#             meta_model_path,
#             static_prob if static_prob else 0.0,
#             process_prob if process_prob else 0.0,
#         )
#         fusion_method = "meta_classifier"
#     else:
#         w_static, w_process = weights
#         final_score = weighted_fusion(static_prob, process_prob, w_static, w_process)
#         fusion_method = "weighted"
#
#     # ---------------------
#     #  Final decision
#     # ---------------------
#     verdict = make_decision(final_score)
#
#     # ---------------------
#     #  Output JSON
#     # ---------------------
#     return {
#         "static": {
#             "present": static_output is not None,
#             "predicted_family": pred_family,
#             "family_confidence": family_conf,
#             "P_malicious": static_prob,
#         },
#         "process": {
#             "present": process_output is not None,
#             "P_malicious": process_prob,
#         },
#         "fusion": {
#             "method": fusion_method,
#             "weights": {"static": weights[0], "process": weights[1]} if fusion_method == "weighted" else None,
#             "final_score": final_score,
#             "verdict": verdict
#         }
#     }
#
#
# # -------------------------------------------------------------
# # CLI FOR STANDALONE TESTING
# # -------------------------------------------------------------
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Fusion Layer CLI")
#
#     parser.add_argument("--static_probs", type=str, help="JSON list of static probabilities")
#     parser.add_argument("--class_names", type=str, help="JSON list of classes")
#     parser.add_argument("--process_prob", type=float, help="Process malicious probability")
#     parser.add_argument("--meta_model_path", type=str, help="Path to meta-classifier")
#     parser.add_argument("--weights", nargs=2, type=float, default=[0.6, 0.4])
#     args = parser.parse_args()
#
#     # Parse static probs if provided
#     static_output = None
#     if args.static_probs:
#         static_output = {
#             "class_probs": json.loads(args.static_probs),
#             "predicted_family": None,
#             "confidence": None
#         }
#
#     class_names = json.loads(args.class_names) if args.class_names else None
#
#     process_output = None
#     if args.process_prob is not None:
#         process_output = {"malicious_prob": args.process_prob}
#
#     result = fuse_predictions(
#         static_output=static_output,
#         process_output=process_output,
#         class_names=class_names,
#         weights=tuple(args.weights),
#         meta_model_path=args.meta_model_path
#     )
#
#     print(json.dumps(result, indent=2))
