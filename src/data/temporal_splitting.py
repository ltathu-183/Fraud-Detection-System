"""
Temporal Train/Validation/Test Splitting for Fraud Detection
=============================================================

Leakage-safe temporal splitting utilities for fraud detection systems.

Key improvements over naive temporal splitting:
- Chronological ordering only
- Optional embargo/gap period to reduce temporal leakage
- Split by timestamp instead of row count
- Stable sorting
- NaN timestamp validation
- Strong validation checks
- Proper logging configuration
- Optional index-only mode for memory efficiency
- Safer temporal cross-validation

IMPORTANT:
TransactionDT in IEEE-CIS is NOT a real timestamp.
It represents elapsed seconds from an arbitrary reference point.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# -----------------------------------------------------------------------------
# Main temporal split
# -----------------------------------------------------------------------------

def temporal_train_val_test_split(
    df: pd.DataFrame,
    time_column: str = "TransactionDT",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    label_column: str = "isFraud",
    embargo_gap: Optional[float] = None,
    return_indices: bool = False
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Perform leakage-safe temporal train/validation/test split.

    This function:
    - sorts chronologically
    - splits by timestamp quantiles
    - optionally inserts embargo gaps
    - prevents direct future leakage

    IMPORTANT:
    This does NOT prevent leakage from:
    - target encoding
    - rolling aggregations
    - global normalization
    - future-aware feature engineering
    """

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    ratios_sum = train_ratio + val_ratio + test_ratio

    if not np.isclose(ratios_sum, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {ratios_sum:.4f}"
        )

    for ratio in [train_ratio, val_ratio, test_ratio]:
        if ratio <= 0 or ratio >= 1:
            raise ValueError(
                "All ratios must be between 0 and 1"
            )

    if time_column not in df.columns:
        raise ValueError(
            f"Missing time column: {time_column}"
        )

    if df[time_column].isna().any():
        raise ValueError(
            f"NaN values detected in {time_column}"
        )

    if embargo_gap is None:
        embargo_gap = 0

    # -------------------------------------------------------------------------
    # Stable chronological sorting
    # -------------------------------------------------------------------------

    df_sorted = df.sort_values(
        time_column,
        kind="mergesort"
    )

    n = len(df_sorted)

    time_values = df_sorted[time_column].values

    # -------------------------------------------------------------------------
    # Quantile-based temporal boundaries
    # -------------------------------------------------------------------------

    train_end_time = df_sorted[time_column].quantile(
        train_ratio
    )

    val_end_time = df_sorted[time_column].quantile(
        train_ratio + val_ratio
    )

    # -------------------------------------------------------------------------
    # Convert temporal boundaries -> indices
    # -------------------------------------------------------------------------

    train_end_idx = np.searchsorted(
        time_values,
        train_end_time,
        side="right"
    )

    val_start_idx = np.searchsorted(
        time_values,
        train_end_time + embargo_gap,
        side="right"
    )

    val_end_idx = np.searchsorted(
        time_values,
        val_end_time,
        side="right"
    )

    test_start_idx = np.searchsorted(
        time_values,
        val_end_time + embargo_gap,
        side="right"
    )

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------

    train_df = df_sorted.iloc[:train_end_idx]
    val_df = df_sorted.iloc[val_start_idx:val_end_idx]
    test_df = df_sorted.iloc[test_start_idx:]

    # -------------------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------------------

    if len(train_df) == 0:
        raise ValueError("Train split is empty")

    if len(val_df) == 0:
        raise ValueError("Validation split is empty")

    if len(test_df) == 0:
        raise ValueError("Test split is empty")

    if (
        train_df[time_column].max()
        >=
        val_df[time_column].min()
    ):
        raise ValueError(
            "Temporal leakage detected between train and validation"
        )

    if (
        val_df[time_column].max()
        >=
        test_df[time_column].min()
    ):
        raise ValueError(
            "Temporal leakage detected between validation and test"
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
    logger.info("=" * 80)

    logger.info(f"Total samples: {n:,}")

    logger.info(
        "NOTE: actual split ratios may differ slightly "
        "due to timestamp grouping."
    )

    logger.info(
        f"Train: {len(train_df):,} "
        f"({len(train_df)/n:.2%})"
    )

    logger.info(
        f"Validation: {len(val_df):,} "
        f"({len(val_df)/n:.2%})"
    )

    logger.info(
        f"Test: {len(test_df):,} "
        f"({len(test_df)/n:.2%})"
    )

    logger.info("")
    logger.info(f"Embargo gap: {embargo_gap}")

    # -------------------------------------------------------------------------
    # Fraud rate monitoring
    # -------------------------------------------------------------------------

    if label_column in df.columns:

        train_rate = train_df[label_column].mean()
        val_rate = val_df[label_column].mean()
        test_rate = test_df[label_column].mean()

        logger.info("")
        logger.info("Fraud rates:")

        logger.info(f"  Train: {train_rate:.4%}")
        logger.info(f"  Validation: {val_rate:.4%}")
        logger.info(f"  Test: {test_rate:.4%}")

        max_shift = max(
            abs(train_rate - val_rate),
            abs(val_rate - test_rate)
        )

        if max_shift > 0.02:
            logger.warning(
                "Large fraud-rate shift detected across splits. "
                "Potential concept drift detected."
            )

    # -------------------------------------------------------------------------
    # Temporal ranges
    # -------------------------------------------------------------------------

    logger.info("")
    logger.info("Temporal ranges:")

    logger.info(
        f"  Train: "
        f"[{train_df[time_column].min():,.0f} -> "
        f"{train_df[time_column].max():,.0f}]"
    )

    logger.info(
        f"  Validation: "
        f"[{val_df[time_column].min():,.0f} -> "
        f"{val_df[time_column].max():,.0f}]"
    )

    logger.info(
        f"  Test: "
        f"[{test_df[time_column].min():,.0f} -> "
        f"{test_df[time_column].max():,.0f}]"
    )

    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------

    if return_indices:
        return (
            train_df.index.values,
            val_df.index.values,
            test_df.index.values
        )

    return (
        train_df.copy(),
        val_df.copy(),
        test_df.copy()
    )


# -----------------------------------------------------------------------------
# Temporal cross-validation
# -----------------------------------------------------------------------------

def temporal_cross_validation_split(
    df: pd.DataFrame,
    time_column: str = "TransactionDT",
    n_splits: int = 5,
    validation_window: float = 0.10,
    embargo_gap: float = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window temporal cross-validation.

    Fold structure:

        Fold 1:
            Train = early history
            Validation = future window

        Fold 2:
            Train = expanded history
            Validation = next future window

    IMPORTANT:
    Splitting is based on temporal ordering,
    NOT random sampling.

    Memory-optimized implementation:
    - NumPy index operations only
    - no repeated .loc calls
    - no repeated pandas object creation
    """

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if time_column not in df.columns:
        raise ValueError(
            f"Missing time column: {time_column}"
        )

    if df[time_column].isna().any():
        raise ValueError(
            f"NaN values detected in {time_column}"
        )

    if validation_window <= 0 or validation_window >= 1:
        raise ValueError(
            "validation_window must be between 0 and 1"
        )

    if embargo_gap < 0:
        raise ValueError(
            "embargo_gap must be >= 0"
        )

    # -------------------------------------------------------------------------
    # Stable sorting
    # -------------------------------------------------------------------------

    df_sorted = df.sort_values(
        time_column,
        kind="mergesort"
    )

    n = len(df_sorted)

    # -------------------------------------------------------------------------
    # Cache NumPy arrays once
    # -------------------------------------------------------------------------

    time_values = (
        df_sorted[time_column]
        .to_numpy(copy=False)
    )

    sorted_indices = (
        df_sorted.index
        .to_numpy(copy=False)
    )

    val_size = int(n * validation_window)

    if val_size == 0:
        raise ValueError(
            "validation_window too small"
        )

    folds = []

    logger.info("=" * 80)
    logger.info("TEMPORAL CROSS-VALIDATION")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Expanding window folds
    # -------------------------------------------------------------------------

    for fold_idx in range(n_splits):

        train_end_idx = val_size * (fold_idx + 1)

        if train_end_idx >= n:
            break

        # ---------------------------------------------------------------------
        # Temporal embargo logic
        # ---------------------------------------------------------------------

        train_end_time = time_values[
            train_end_idx - 1
        ]

        val_start_time = (
            train_end_time +
            embargo_gap
        )

        val_start_idx = np.searchsorted(
            time_values,
            val_start_time,
            side="right"
        )

        val_end_idx = (
            val_start_idx +
            val_size
        )

        if val_end_idx > n:
            break

        # ---------------------------------------------------------------------
        # NumPy-only index slicing
        # ---------------------------------------------------------------------

        train_indices = sorted_indices[
            :train_end_idx
        ]

        val_indices = sorted_indices[
            val_start_idx:val_end_idx
        ]

        if train_indices.size == 0:
            continue

        if val_indices.size == 0:
            continue

        # ---------------------------------------------------------------------
        # Leakage safety check
        # ---------------------------------------------------------------------

        train_max_time = time_values[
            train_end_idx - 1
        ]

        val_min_time = time_values[
            val_start_idx
        ]

        if train_max_time >= val_min_time:
            raise ValueError(
                f"Temporal leakage detected in fold "
                f"{fold_idx + 1}"
            )

        folds.append(
            (
                train_indices,
                val_indices
            )
        )

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------

        logger.info(
            f"Fold {fold_idx + 1}: "
            f"train={train_indices.size:,}, "
            f"validation={val_indices.size:,}"
        )

        logger.info(
            f"  Train range: "
            f"[{time_values[0]:,.0f} -> "
            f"{train_max_time:,.0f}]"
        )

        logger.info(
            f"  Validation range: "
            f"[{val_min_time:,.0f} -> "
            f"{time_values[val_end_idx-1]:,.0f}]"
        )

    logger.info("=" * 80)

    if len(folds) == 0:
        raise ValueError(
            "No valid folds generated. "
            "Try reducing n_splits or validation_window."
        )

    return folds