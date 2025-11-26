# src_process/eval_process.py
import argparse, joblib, os, numpy as np, pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score

def evaluate(model_path, scaler_path, csv_path, label_col=None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(csv_path)

    # detect label
    if label_col and label_col in df.columns:
        labcol = label_col
    else:
        labcol = 'Class' if 'Class' in df.columns else ('Category' if 'Category' in df.columns else None)
        if labcol is None:
            # fallback: first column that looks like label
            for c in df.columns:
                vals = set([str(v).upper() for v in df[c].dropna().unique()[:10]])
                if 'BENIGN' in vals or 'MALICIOUS' in vals:
                    labcol = c
                    break
    if labcol is None:
        raise ValueError("Could not find label column. Pass --label_col explicitly.")

    y_raw = df[labcol]
    if y_raw.dtype == object or y_raw.dtype.name.startswith('str'):
        y = y_raw.astype(str).str.upper().map({'BENIGN':0,'MALICIOUS':1,'MALWARE':1}).fillna(0).astype(int)
    else:
        y = y_raw.fillna(0).astype(int)

    # features: use numeric columns except the label
    X = df.select_dtypes(include=[np.number]).drop(columns=[labcol], errors='ignore')
    X = X.fillna(0)
    X_scaled = scaler.transform(X.values)

    preds = model.predict(X_scaled)
    try:
        probs = model.predict_proba(X_scaled)[:,1]
    except Exception:
        probs = None

    print("Accuracy:", accuracy_score(y, preds))
    print("Macro-F1:", f1_score(y, preds, average='macro'))
    print(classification_report(y, preds))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--scaler_path", required=True)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--label_col", default=None)
    args = p.parse_args()
    evaluate(args.model_path, args.scaler_path, args.csv_path, label_col=args.label_col)
