# src_process/dataset_process.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_CANDIDATES = ['Class', 'class', 'ClassName', 'Label', 'label', 'Category', 'category']

def find_label_column(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: any column with string values 'Benign'/'Malicious'
    for c in df.columns:
        vals = df[c].dropna().unique()
        sval = set([str(v).strip().upper() for v in vals[:10]])
        if 'BENIGN' in sval or 'MALICIOUS' in sval or 'MALWARE' in sval:
            return c
    raise ValueError("Label column not found. Please ensure 'Class' or 'Category' exists or pass label_col.")

def select_features(df, label_col, blacklist_keywords=None):
    # In your dataset most columns are numeric; pick numeric cols and exclude obvious identifiers
    blacklist_keywords = blacklist_keywords or ['id','name','path','file','hash','time','timestamp','date']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = [c for c in numeric_cols if all(b not in c.lower() for b in blacklist_keywords) and c != label_col]
    # If dataset already uses mostly numeric columns and selected ends empty, fallback to all numeric
    if not selected:
        selected = numeric_cols
    return selected

def load_and_prepare(csv_path, label_col=None, test_size=0.2, random_state=42, stratify=True, verbose=True):
    """
    Loads the CICMalMem CSV that follows the schema you provided.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_columns, label_col_used
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    if verbose:
        print(f"[dataset] CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # detect label
    labcol = label_col if (label_col and label_col in df.columns) else find_label_column(df)
    if verbose:
        print(f"[dataset] using label column: {labcol}")

    # normalize label values to 0/1: BENIGN->0, others->1
    y_raw = df[labcol].copy()
    if y_raw.dtype == object or y_raw.dtype.name.startswith('str'):
        y = y_raw.astype(str).str.upper().map({'BENIGN':0,'MALICIOUS':1,'MALWARE':1}).fillna(-1).astype(int)
    else:
        y = y_raw.fillna(0).astype(int)

    # feature selection - pick numeric columns excluding label
    feature_cols = select_features(df, labcol)
    if verbose:
        print(f"[dataset] selected {len(feature_cols)} numeric features")

    X = df[feature_cols].copy()
    # cleanup
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    stratify_arg = y if (stratify and len(np.unique(y)) > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=random_state, stratify=stratify_arg)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols, labcol
