# src_process/model_process.py
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    XGBClassifier = None
    HAS_XGB = False

def get_model(name='rf', random_state=42, class_weight=None, n_jobs=-1):
    name = name.lower()
    if name == 'rf':
        return RandomForestClassifier(n_estimators=200, max_depth=None, random_state=random_state, class_weight=class_weight, n_jobs=n_jobs)
    elif name == 'xgb':
        if not HAS_XGB:
            raise ImportError("xgboost not installed. Install with `pip install xgboost` or choose 'rf'.")
        return XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, random_state=random_state, n_jobs=n_jobs,
                             use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("model must be 'rf' or 'xgb'")
