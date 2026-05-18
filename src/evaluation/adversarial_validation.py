"""
Adversarial validation module for temporal fraud pipeline.

This module compares training distribution against later splits
without leaking target labels into the main model.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def run_adversarial_validation(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: List[str],
    sample_size: Optional[int] = 20000,
    random_state: int = 42
) -> Dict[str, object]:
    """
    Train a lightweight adversarial classifier to detect split drift.

    Parameters
    ----------
    source_df : pd.DataFrame
        Source split (e.g. train).
    target_df : pd.DataFrame
        Target split to compare (e.g. val or test).
    feature_cols : List[str]
        Numeric features to use for drift detection.
    sample_size : Optional[int]
        Maximum number of rows per split to sample for efficiency.
    random_state : int
        Random seed.
    """

    if sample_size is not None:
        source_df = source_df.sample(
            n=min(sample_size, len(source_df)),
            random_state=random_state
        )
        target_df = target_df.sample(
            n=min(sample_size, len(target_df)),
            random_state=random_state
        )

    X_source = source_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X_target = target_df[feature_cols].to_numpy(dtype=np.float32, copy=False)

    X = np.vstack([X_source, X_target])
    y = np.concatenate([
        np.zeros(len(X_source), dtype=np.int8),
        np.ones(len(X_target), dtype=np.int8)
    ])

    if len(feature_cols) == 0:
        raise ValueError("No numeric features available for adversarial validation.")

    clf = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        max_iter=500,
        random_state=random_state
    )

    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    roc_auc = float(roc_auc_score(y, probs))

    coef = np.abs(clf.coef_[0])
    top_features = sorted(
        zip(feature_cols, coef),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return {
        'roc_auc': roc_auc,
        'samples': {
            'source': len(X_source),
            'target': len(X_target)
        },
        'top_drift_features': [
            {'feature': name, 'weight': float(weight)}
            for name, weight in top_features
        ]
    }
