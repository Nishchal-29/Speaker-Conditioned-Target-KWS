"""
Equal Error Rate (EER) computation utility.

Used by:
  - Context A: Training validation loop + ONNX export validation (50-pair check)
  - Context B: (indirectly, for offline analysis of enrollment quality)
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(scores, labels):
    """
    Computes the Equal Error Rate (EER) from similarity scores and binary labels.

    Args:
        scores: array-like of float — cosine similarity scores for each pair.
        labels: array-like of int/bool — 1 = same speaker, 0 = different speaker.

    Returns:
        eer: float — the Equal Error Rate (0.0 to 1.0).
        threshold: float — the similarity threshold at which FAR == FRR.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    if len(scores) != len(labels):
        raise ValueError(f"scores ({len(scores)}) and labels ({len(labels)}) must have the same length")

    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("labels must contain at least one positive (1) and one negative (0) example")

    # Sort by descending score to sweep thresholds from high to low
    sorted_indices = np.argsort(-scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Total number of genuine (same-speaker) and impostor (different-speaker) pairs
    n_genuine = np.sum(labels == 1)
    n_impostor = np.sum(labels == 0)

    # Compute FAR and FRR at each threshold
    # FAR  = fraction of impostor pairs accepted (score >= threshold)
    # FRR  = fraction of genuine pairs rejected  (score < threshold)
    fars = []
    frrs = []
    thresholds = []

    for i in range(len(sorted_scores)):
        threshold = sorted_scores[i]

        # Everything at index <= i has score >= threshold (accepted)
        accepted_labels = sorted_labels[:i + 1]
        rejected_labels = sorted_labels[i + 1:]

        false_accepts = np.sum(accepted_labels == 0)
        false_rejects = np.sum(rejected_labels == 1)

        far = false_accepts / n_impostor
        frr = false_rejects / n_genuine

        fars.append(far)
        frrs.append(frr)
        thresholds.append(threshold)

    fars = np.array(fars)
    frrs = np.array(frrs)
    thresholds = np.array(thresholds)

    # Find EER: the point where FAR == FRR via interpolation
    try:
        eer = brentq(
            lambda x: interp1d(thresholds, fars)(x) - interp1d(thresholds, frrs)(x),
            thresholds[-1],  # lowest threshold
            thresholds[0],   # highest threshold
        )
        eer_value = float(interp1d(thresholds, fars)(eer))
    except ValueError:
        # Fallback: find the threshold with minimum |FAR - FRR|
        abs_diff = np.abs(fars - frrs)
        min_idx = np.argmin(abs_diff)
        eer = thresholds[min_idx]
        eer_value = (fars[min_idx] + frrs[min_idx]) / 2.0

    return eer_value, eer
