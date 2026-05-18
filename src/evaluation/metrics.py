"""
Evaluation Metrics for Fraud Detection
=======================================

Following IEEE-CIS blueprint: STEP 14 & 15

PRIMARY: PR-AUC (Precision-Recall AUC)
Best metric for severe class imbalance.

SECONDARY: ROC-AUC, Recall, Precision, F1, MCC

BUSINESS METRICS:
- Fraud capture rate
- False decline rate
- Review rate

⚠️ USAGE GUIDE:
- For MODEL/RANKING evaluation: use calculate_metrics() or calculate_ranking_metrics() with PROBABILITIES
- For DECISION/SYSTEM evaluation: use evaluate_three_tier_system() or evaluate_3tier_decision() with PROBABILITIES + thresholds
- NEVER compute AUC on binary predictions (y_pred) - only on probabilities (y_pred_proba)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_curve, roc_curve,
    f1_score, confusion_matrix, matthews_corrcoef
)
from typing import Dict, Tuple, Union
import logging

logger = logging.getLogger(__name__)


# =========================================================
# EXISTING FUNCTIONS (UNCHANGED - PRESERVE COMPATIBILITY)
# =========================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics at a specific single threshold.
    
    ⚠️ IMPORTANT: 
    - y_pred_proba should be RAW PROBABILITIES from model.predict_proba()[:, 1]
    - AUC metrics (roc_auc, pr_auc) are computed on probabilities, NOT binary predictions
    - Binary metrics (precision, recall, f1) are computed at the given threshold
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Primary metric: PR-AUC (best for imbalance)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    metrics = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'threshold': threshold,
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    
    # Per-class metrics
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0  
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Business metrics
    total = len(y_true)
    metrics['fraud_capture_rate'] = metrics['recall']  
    metrics['false_decline_rate'] = fp / (tn + fp) if (tn + fp) > 0 else 0  # FP / Total Legitimate
    metrics['approval_rate'] = (tn + fn) / total  
    metrics['review_rate'] = (fp + tp) / total  
    
    return metrics


def rolling_time_validation(
    df: pd.DataFrame,
    n_folds: int = 3,
    label_col: str = 'isFraud',
    params: Dict = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 25
) -> Dict[str, object]:
    """
    Perform expanding-window rolling time validation on chronological data.

    Returns fold-level metrics plus average and standard deviation.
    """
    import lightgbm as lgb

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")

    df = df.reset_index(drop=True)
    n = len(df)
    if n < 100:
        raise ValueError("Data too small for rolling time validation.")

    feature_cols = [
        col for col in df.columns
        if col not in {label_col, 'TransactionID', 'TransactionDT'}
        and np.issubdtype(df[col].dtype, np.number)
    ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric features available for rolling time validation.")

    params = params or {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 48,
        'max_depth': 6,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'min_child_samples': 200,
        'lambda_l1': 1.0,
        'lambda_l2': 2.0,
        'verbose': -1,
        'seed': 42,
        'deterministic': True
    }

    fold_metrics = []
    split_step = 0.1
    base_train_frac = 0.5
    val_frac = 0.1

    for fold in range(n_folds):
        train_end = int(n * (base_train_frac + fold * split_step))
        val_end = int(n * (base_train_frac + fold * split_step + val_frac))

        if val_end > n:
            break

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]

        dtrain = lgb.Dataset(
            train_df[feature_cols],
            label=train_df[label_col].to_numpy(),
            free_raw_data=False
        )
        dvalid = lgb.Dataset(
            val_df[feature_cols],
            label=val_df[label_col].to_numpy(),
            reference=dtrain,
            free_raw_data=False
        )

        booster = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=[dvalid],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )

        y_val_pred = booster.predict(
            val_df[feature_cols],
            num_iteration=booster.best_iteration
        )

        fold_result = calculate_metrics(
            val_df[label_col].to_numpy(),
            y_val_pred,
            threshold=0.5
        )
        fold_result.update({
            'fold': fold + 1,
            'train_end': train_end,
            'val_end': val_end,
            'best_iteration': int(booster.best_iteration)
        })
        fold_metrics.append(fold_result)

    aggregate = {}
    if fold_metrics:
        keys = [k for k in fold_metrics[0].keys() if k not in {'fold', 'train_end', 'val_end', 'best_iteration'}]
        for key in keys:
            values = np.array([fm[key] for fm in fold_metrics])
            aggregate[f'{key}_mean'] = float(values.mean())
            aggregate[f'{key}_std'] = float(values.std())

    return {
        'fold_metrics': fold_metrics,
        'aggregate': aggregate
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = 'f_beta',
    beta: float = 2,
    threshold_range: tuple = (0.01, 0.99),
    n_steps: int = 100
) -> Tuple[float, float]:
    """
    Find optimal single threshold for specified metric.
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    best_threshold = 0.5
    best_score = -np.inf
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'f_beta':
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            beta_sq = beta ** 2
            score = (1 + beta_sq) * (precision * recall) / (
                (beta_sq * precision) + recall + 1e-10
            )
        elif metric == 'precision':
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            score = tp / (tp + fp) if (tp + fp) > 0 else 0
        elif metric == 'recall':
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            score = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def optimize_three_tier_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    target_recall: float = 0.85,
    max_false_decline_rate: float = 0.02
) -> Tuple[float, float]:
    """
    Optimize thresholds for three-tier decision system.

    Business interpretation:
    - score < low_threshold: APPROVE (low risk)
    - low_threshold <= score < high_threshold: MANUAL REVIEW (medium risk)
    - score >= high_threshold: DECLINE/BLOCK (high risk)
    """
    calibrated_probs = np.clip(
        y_pred_proba,
        1e-8,
        1.0 - 1e-8
    )

    unique_thresholds = np.unique(np.sort(calibrated_probs))
    if len(unique_thresholds) < 2:
        return 0.01, 0.99

    if len(unique_thresholds) > 200:
        unique_thresholds = np.unique(
            np.quantile(
                unique_thresholds,
                np.linspace(0.0, 1.0, 200)
            )
        )

    fraud_missed_cost = 10.0
    false_decline_cost = 1.0
    review_cost = 0.25

    best_score = -np.inf
    best_low = 0.01
    best_high = 0.99

    total_legit = int((y_true == 0).sum())
    if total_legit == 0:
        raise ValueError("No negative examples available for three-tier threshold optimization.")

    for i in range(len(unique_thresholds) - 1):
        low = float(unique_thresholds[i])

        for j in range(i + 1, len(unique_thresholds)):
            high = float(unique_thresholds[j])

            if low >= high:
                continue

            # masks (compute ONCE)
            y_approve = calibrated_probs < low
            y_decline = calibrated_probs >= high
            y_review = ~(y_approve | y_decline)

            # counts
            fp = int(((y_true == 0) & y_decline).sum())
            fn = int(((y_true == 1) & ~y_decline).sum())

            review_count = int(y_review.sum())
            approve_rate = float(y_approve.mean())

            total_legit = (y_true == 0).sum()
            false_decline = fp / total_legit if total_legit > 0 else 0.0

            # =========================
            # HARD BUSINESS CONSTRAINTS
            # =========================

            if false_decline > max_false_decline_rate:
                continue

            if approve_rate > 0.95:
                continue

            if review_count / len(y_true) > 0.60:
                continue

            # =========================
            # COST FUNCTION
            # =========================
            cost = (
                fn * fraud_missed_cost +
                fp * false_decline_cost +
                review_count * review_cost
            )

            utility = -cost

            if utility > best_score:
                best_score = utility
                best_low = low
                best_high = high

    if best_score == -np.inf:
        logger.warning(
            "No valid three-tier candidate passed false decline constraints. "
            "Using safe fallback thresholds."
        )
        best_low = float(np.percentile(calibrated_probs, 0.10))
        best_high = float(np.percentile(calibrated_probs, 0.90))

    best_low = max(0.0, min(best_low, 0.99))
    best_high = max(best_low + 1e-3, min(best_high, 0.99))

    assert 0.0 <= best_low < best_high <= 1.0, (
        f"Invalid thresholds after optimization: low={best_low}, high={best_high}"
    )

    logger.info("\nThree-Tier Thresholds Optimized Successfully:")
    logger.info(f"   Low threshold (approve boundary) : {best_low:.4f}")
    logger.info(f"   High threshold (decline boundary): {best_high:.4f}")
    logger.info(f"   Target recall: {target_recall:.2%}")
    logger.info(f"   Max false decline limit: {max_false_decline_rate:.2%}")

    return best_low, best_high


def evaluate_three_tier_system(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    low_threshold: float = 0.2,
    high_threshold: float = 0.7
) -> Dict[str, float]:

    assert 0.0 <= low_threshold < high_threshold <= 1.0

    y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)

    approve = y_pred_proba < low_threshold
    review = (y_pred_proba >= low_threshold) & (y_pred_proba < high_threshold)
    decline = y_pred_proba >= high_threshold

    total = len(y_true)
    total_fraud = max((y_true == 1).sum(), 1)
    total_legit = max((y_true == 0).sum(), 1)

    # fraud distribution
    fraud_in_approve = int(((y_true == 1) & approve).sum())
    fraud_in_review = int(((y_true == 1) & review).sum())
    fraud_in_decline = int(((y_true == 1) & decline).sum())

    legit_in_decline = int(((y_true == 0) & decline).sum())
    legit_in_review = int(((y_true == 0) & review).sum())

    metrics = {
        "approve_rate": float(approve.mean()),
        "review_rate": float(review.mean()),
        "decline_rate": float(decline.mean()),

        "fraud_capture_rate": float(fraud_in_decline / total_fraud),

        "false_decline_rate": float(legit_in_decline / total_legit),

        "review_legit_rate": float(legit_in_review / total_legit),

        "fraud_in_approve": fraud_in_approve,
        "fraud_in_review": fraud_in_review,
        "fraud_in_decline": fraud_in_decline,
    }

    # safety invariants
    assert abs(metrics["approve_rate"] + metrics["review_rate"] + metrics["decline_rate"] - 1.0) < 1e-6

    return metrics


# =========================================================
# NEW HELPER FUNCTIONS (NON-BREAKING ADDITIONS)
# =========================================================

def calculate_ranking_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    [NEW] Evaluate MODEL RANKING performance only.
    
    Use this for: model selection, AUC reporting, probability calibration checks.
    
    Input: 
        - y_true: ground truth labels (0/1)
        - y_pred_proba: RAW probabilities from model.predict_proba()[:, 1]
    
    Output:
        - roc_auc, pr_auc, and probability distribution stats
        - NO binary metrics (no threshold applied)
    
    ✅ This is what you should use to verify model quality (e.g., ROC-AUC ~0.88)
    """
    return {
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_true, y_pred_proba)),
        'mean_prob': float(np.mean(y_pred_proba)),
        'std_prob': float(np.std(y_pred_proba)),
        'min_prob': float(np.min(y_pred_proba)),
        'max_prob': float(np.max(y_pred_proba))
    }


def evaluate_3tier_decision(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    include_breakdown: bool = True
) -> Dict[str, Union[float, int]]:
    """
    [NEW] Evaluate 3-TIER DECISION SYSTEM performance.
    
    Use this for: business metrics, operational reporting, threshold tuning.
    
    Input:
        - y_true: ground truth labels (0/1)
        - y_pred_proba: RAW probabilities (NOT binary predictions)
        - low_threshold: approve/review boundary
        - high_threshold: review/block boundary
    
    Output:
        - All metrics from evaluate_three_tier_system() PLUS:
        - recall_total: fraud captured via block + review (most important for fraud)
        - utility_score: simple business utility estimate
        - decision_counts: raw counts for audit
    
    ✅ This is what you should use for final system evaluation (NOT calculate_metrics on binary)
    """
    # Reuse existing function for core metrics
    base_metrics = evaluate_three_tier_system(
        y_true, y_pred_proba, low_threshold, high_threshold
    )
    
    total_fraud = max((y_true == 1).sum(), 1)
    
    # Add recall_total: fraud caught by block OR review (not just block)
    fraud_caught = base_metrics['fraud_in_decline'] + base_metrics['fraud_in_review']
    base_metrics['recall_total'] = fraud_caught / total_fraud
    
    # Add simple utility score (configurable weights)
    utility = (
        base_metrics['fraud_in_decline'] * 10.0 +      # TP: blocked fraud
        base_metrics['fraud_in_review'] * 2.0 +        # Partial: reviewed fraud  
        - base_metrics['false_decline_rate'] * 100 * 2.0 +  # FP cost
        - base_metrics['review_rate'] * 100 * 0.5      # Review cost
    )
    base_metrics['utility_score'] = float(utility)
    
    if include_breakdown:
        base_metrics.update({
            'decision_counts': {
                'approve': int((y_pred_proba < low_threshold).sum()),
                'review': int(((y_pred_proba >= low_threshold) & (y_pred_proba < high_threshold)).sum()),
                'block': int((y_pred_proba >= high_threshold).sum()),
                'total': len(y_true)
            }
        })
    
    return base_metrics


def validate_evaluation_inputs(
    y_true: np.ndarray,
    y_pred_input: np.ndarray,
    context: str = "metrics calculation"
) -> None:
    """
    [NEW] Helper to catch common evaluation mistakes early.
    
    Use this at the start of your evaluation code to prevent silent bugs.
    
    Examples:
        validate_evaluation_inputs(y_test, test_probs, "test set ranking eval")
        validate_evaluation_inputs(y_val, val_probs, "threshold optimization")
    """
    if not isinstance(y_pred_input, np.ndarray):
        raise TypeError(f"[{context}] y_pred must be numpy array, got {type(y_pred_input)}")
    
    if y_pred_input.dtype not in [np.float32, np.float64, np.float16]:
        # Likely binary predictions passed where probs expected
        unique_vals = np.unique(y_pred_input)
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            logger.warning(
                f"[{context}] ⚠️ y_pred appears to be binary (values: {unique_vals}). "
                f"AUC metrics require PROBABILITIES. Did you pass y_pred instead of y_pred_proba?"
            )
    
    if np.all((y_pred_input >= 0) & (y_pred_input <= 1)):
        if np.all(np.isin(y_pred_input, [0, 1])):
            logger.warning(
                f"[{context}] ⚠️ All probabilities are exactly 0 or 1. "
                f"This suggests binary predictions were passed. AUC will be meaningless."
            )
    
    if len(y_true) != len(y_pred_input):
        raise ValueError(f"[{context}] Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred_input)}")
    
    logger.debug(f"[{context}] ✓ Input validation passed: n={len(y_true)}, prob_range=[{y_pred_input.min():.3f}, {y_pred_input.max():.3f}]")