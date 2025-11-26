# src/ensemble.py
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def weighted_soft_vote(df1_path, df2_path, w1=0.5, w2=0.5):
    df1 = pd.read_csv(df1_path)  # has p_0 ... p_C-1
    df2 = pd.read_csv(df2_path)
    # ensure rows align (they should, if same test set and order)
    pcols = [c for c in df1.columns if c.startswith('p_')]
    P1 = df1[pcols].values
    P2 = df2[pcols].values
    P = w1*P1 + w2*P2
    preds = P.argmax(axis=1)
    y_true = df1['true'].values
    f1 = f1_score(y_true, preds, average='macro')
    print("Ensemble macro-F1:", f1)
    return preds, f1

# example usage:
# preds, f1 = weighted_soft_vote('models/resnet50_run1/resnet50_test_probs_expanded.csv',
#                                'models/eff_run1/efficientnet_test_probs_expanded.csv',
#                                w1=0.4, w2=0.6)
