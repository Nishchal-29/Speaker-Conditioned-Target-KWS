import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]

    return float(eer), float(threshold)