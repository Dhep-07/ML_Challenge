"""
F_beta (e.g. F_0.5) optimized threshold search for precision-skewed classification.
"""
from sklearn.metrics import fbeta_score
import numpy as np

def tune_threshold(y_true, y_probs, beta=0.5):
    """Searches for decision threshold maximizing F_beta score."""
    best_threshold = 0.5
    best_score = 0.0
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (y_probs >= threshold).astype(int)
        score = fbeta_score(y_true, preds, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score
