# src_process/train_process.py
import argparse, os, joblib, numpy as np, pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from dataset_process import load_and_prepare
from model_process import get_model

def train(csv_path, out_dir, model_name='rf', test_size=0.2, label_col=None, use_class_weight=False, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, scaler, feature_cols, label_used = load_and_prepare(csv_path, label_col=label_col, test_size=test_size, random_state=random_state)

    class_weight = 'balanced' if use_class_weight else None
    model = get_model(model_name, random_state=random_state, class_weight=class_weight)

    print(f"[train] training {model_name} on {X_train.shape[0]} samples with {len(feature_cols)} features")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    try:
        probs = model.predict_proba(X_test)[:,1]
    except Exception:
        probs = None

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro') if len(np.unique(y_test))>1 else f1_score(y_test, preds)
    print(f"[train] Test accuracy={acc:.4f}, macro-F1={macro_f1:.4f}")
    print(classification_report(y_test, preds))

    # Save artifacts
    model_path = os.path.join(out_dir, f"{model_name}_process.pkl")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    features_path = os.path.join(out_dir, "feature_columns.txt")
    metrics_path = os.path.join(out_dir, "metrics.csv")
    cm_path = os.path.join(out_dir, "confusion_matrix.csv")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    pd.Series(feature_cols).to_csv(features_path, index=False, header=False)
    pd.DataFrame([{"accuracy": acc, "macro_f1": macro_f1, "n_train": X_train.shape[0], "n_test": X_test.shape[0]}]).to_csv(metrics_path, index=False)

    cm = confusion_matrix(y_test, preds)
    pd.DataFrame(cm).to_csv(cm_path, index=False)
    print(f"[train] saved model -> {model_path}")
    print(f"[train] saved scaler -> {scaler_path}")
    print(f"[train] saved feature list -> {features_path}")
    return model_path, scaler_path, features_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True, help="path to CICMalMem2022.csv")
    p.add_argument("--out_dir", default="./models")
    p.add_argument("--model", choices=['rf','xgb'], default='rf')
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--label_col", default=None, help="optional label column name if not auto-detected")
    p.add_argument("--use_class_weight", action='store_true')
    args = p.parse_args()
    train(args.csv_path, args.out_dir, model_name=args.model, test_size=args.test_size, label_col=args.label_col, use_class_weight=args.use_class_weight)
