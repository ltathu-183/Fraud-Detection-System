"""
Feature Engineering for Fraud Detection
========================================

Optimized fraud-focused feature engineering pipeline.

FIXES APPLIED
=============
1. FIXED interaction feature leakage
   - interaction mappings are fit on TRAIN ONLY
   - val/test unknown combinations -> -1

2. FIXED repeated expensive groupby operations
   - groupby stats computed ONCE
   - reused across all datasets

3. FIXED string/object memory blowups
   - removed giant temporary object columns
   - replaced with categorical integer codes
   - uses int32 / float32 aggressively

4. FIXED unnecessary dataframe concatenations
   - no more concat(train,val,test)
   - avoids huge memory spikes

5. FIXED inplace feature engineering inefficiency
   - reduced copies
   - reduced intermediate allocations

IMPORTANT PIPELINE NOTES
========================
- Function names are unchanged
- Return signatures are unchanged
- Feature names are unchanged where possible
- Interaction feature values WILL change
  because they are now leakage-safe
- Aggregation logic changed:
    OLD -> rolling windows
    NEW -> historical train statistics
- This is MUCH faster and production-safe
"""

import gc
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple
from sklearn.preprocessing import OrdinalEncoder


# =============================================================================
# CATEGORICAL ENCODING
# =============================================================================

def encode_categorical_features(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    cat_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Ordinal encode categorical features using TRAIN ONLY.
    """

    if cat_cols is None:
        cat_cols = train_data.select_dtypes(include=['object']).columns.tolist()

    encoders = {}

    print(f"\n{'='*60}")
    print("CATEGORICAL ENCODING")
    print(f"{'='*60}")
    print(f"Encoding {len(cat_cols)} categorical columns")

    for col in cat_cols:

        if col not in train_data.columns:
            continue

        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=np.int32
        )

        train_values = train_data[[col]].fillna('MISSING').astype(str)
        val_values = val_data[[col]].fillna('MISSING').astype(str)
        test_values = test_data[[col]].fillna('MISSING').astype(str)

        encoder.fit(train_values)

        train_data[col] = (
            encoder.transform(train_values)
            .astype(np.int32)
            .ravel()
        )

        val_data[col] = (
            encoder.transform(val_values)
            .astype(np.int32)
            .ravel()
        )

        test_data[col] = (
            encoder.transform(test_values)
            .astype(np.int32)
            .ravel()
        )

        encoders[col] = encoder

        print(
            f"  {col}: "
            f"{len(encoder.categories_[0]):,} categories"
        )

        del train_values
        del val_values
        del test_values
        gc.collect()

    print(f"Encoded {len(encoders)} categorical features")

    return train_data, val_data, test_data, encoders


# =============================================================================
# TIME FEATURES
# =============================================================================

def create_time_features(
    data: pd.DataFrame,
    time_col: str = 'TransactionDT'
) -> pd.DataFrame:
    """
    Extract cyclic time features.
    """

    print("Creating time features...")

    hours = ((data[time_col] // 3600) % 24).astype(np.int8)
    days = ((data[time_col] // (3600 * 24)) % 7).astype(np.int8)

    data['hour_of_day'] = hours
    data['day_of_week'] = days

    data['hour_sin'] = np.sin(
        2 * np.pi * hours / 24
    ).astype(np.float32)

    data['hour_cos'] = np.cos(
        2 * np.pi * hours / 24
    ).astype(np.float32)

    data['day_sin'] = np.sin(
        2 * np.pi * days / 7
    ).astype(np.float32)

    data['day_cos'] = np.cos(
        2 * np.pi * days / 7
    ).astype(np.float32)

    print(
        "Created: hour_of_day, day_of_week, "
        "hour_sin, hour_cos, day_sin, day_cos"
    )

    return data


# =============================================================================
# AMOUNT FEATURES
# =============================================================================

def create_amount_features(
    data: pd.DataFrame,
    amount_col: str = 'TransactionAmt'
) -> pd.DataFrame:
    """
    Create amount-based features.
    """

    print("Creating amount features...")

    amount_values = data[amount_col].astype(np.float32)

    data[f'{amount_col}_log'] = np.log1p(
        amount_values
    ).astype(np.float32)

    data[f'{amount_col}_bin'] = pd.qcut(
        amount_values.rank(method='first'),
        q=10,
        labels=False,
        duplicates='drop'
    ).astype(np.int8)

    print(
        f"Created: {amount_col}_log, "
        f"{amount_col}_bin"
    )

    return data


# =============================================================================
# AGGREGATION FEATURES
# =============================================================================

def create_aggregation_features(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    entity_cols: List[str] = ['card1', 'card2', 'addr1'],
    agg_cols: List[str] = ['TransactionAmt'],
    time_col: str = 'TransactionDT',
    windows: List[int] = [1, 24, 168]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fast aggregation features using TRAIN statistics only.
    """

    print(f"\n{'='*60}")
    print("AGGREGATION FEATURES")
    print(f"{'='*60}")

    for entity_col in entity_cols:

        if entity_col not in train_data.columns:
            continue

        print(f"\nEntity: {entity_col}")

        for agg_col in agg_cols:

            if agg_col not in train_data.columns:
                continue

            print(f"  Aggregating: {agg_col}")

            # ==========================================================
            # SINGLE GROUPBY
            # ==========================================================

            stats = (
                train_data
                .groupby(entity_col)[agg_col]
                .agg(['mean', 'std', 'median', 'count', 'max', 'min'])
            )

            stats.columns = [
                'mean',
                'std',
                'median',
                'count',
                'max',
                'min'
            ]

            for df_name, df in [
                ('train', train_data),
                ('val', val_data),
                ('test', test_data)
            ]:

                mapped = df[[entity_col]].merge(
                    stats,
                    left_on=entity_col,
                    right_index=True,
                    how='left'
                )

                df[f'{entity_col}_{agg_col}_mean'] = (
                    mapped['mean']
                    .astype(np.float32)
                )

                df[f'{entity_col}_{agg_col}_std'] = (
                    mapped['std']
                    .fillna(0)
                    .astype(np.float32)
                )

                df[f'{entity_col}_{agg_col}_median'] = (
                    mapped['median']
                    .astype(np.float32)
                )

                df[f'{entity_col}_{agg_col}_count'] = (
                    mapped['count']
                    .fillna(0)
                    .astype(np.int32)
                )

                df[f'{entity_col}_{agg_col}_max'] = (
                    mapped['max']
                    .astype(np.float32)
                )

                df[f'{entity_col}_{agg_col}_min'] = (
                    mapped['min']
                    .astype(np.float32)
                )

                print(f"    Applied to {df_name}")

                del mapped
                gc.collect()

            del stats
            gc.collect()

    print("Aggregation features completed")

    return train_data, val_data, test_data


# =============================================================================
# DEVIATION FEATURES
# =============================================================================

def create_deviation_features(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    entity_cols: List[str] = ['card1', 'card2'],
    agg_cols: List[str] = ['TransactionAmt']
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create deviation features from historical statistics.
    """

    print(f"\n{'='*60}")
    print("DEVIATION FEATURES")
    print(f"{'='*60}")

    for entity_col in entity_cols:

        if entity_col not in train_data.columns:
            continue

        for agg_col in agg_cols:

            if agg_col not in train_data.columns:
                continue

            print(f"Processing: {entity_col} x {agg_col}")

            # ==========================================================
            # SINGLE GROUPBY
            # ==========================================================

            stats = (
                train_data
                .groupby(entity_col)[agg_col]
                .agg(['mean', 'std', 'median'])
            )

            stats.columns = [
                'mean',
                'std',
                'median'
            ]

            for df in [train_data, val_data, test_data]:

                mapped = df[[entity_col]].merge(
                    stats,
                    left_on=entity_col,
                    right_index=True,
                    how='left'
                )

                hist_mean = mapped['mean']
                hist_std = mapped['std']
                hist_median = mapped['median']

                ratio_feature = (
                    df[agg_col] / (hist_mean + 1e-6)
                )

                zscore_feature = (
                    (df[agg_col] - hist_mean) /
                    (hist_std + 1e-6)
                )

                median_ratio_feature = (
                    df[agg_col] / (hist_median + 1e-6)
                )

                df[f'{entity_col}_{agg_col}_ratio_to_mean'] = (
                    ratio_feature
                    .replace([np.inf, -np.inf], 1)
                    .fillna(1)
                    .astype(np.float32)
                )

                df[f'{entity_col}_{agg_col}_zscore'] = (
                    zscore_feature
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                    .astype(np.float32)
                )

                df[f'{entity_col}_{agg_col}_ratio_to_median'] = (
                    median_ratio_feature
                    .replace([np.inf, -np.inf], 1)
                    .fillna(1)
                    .astype(np.float32)
                )

                del mapped
                gc.collect()

            del stats
            gc.collect()

            print(
                f"  Created: ratio_to_mean, "
                f"zscore, ratio_to_median"
            )

    return train_data, val_data, test_data


# =============================================================================
# INTERACTION FEATURES
# =============================================================================

def create_interaction_features(
    data: pd.DataFrame,
    interaction_pairs: List[Tuple[str, str]] = None,
    fitted_mappings: Dict = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Create leakage-safe interaction features.

    IMPORTANT:
    ----------
    - TRAIN fits mappings
    - VAL/TEST reuse mappings
    - Unknown combos -> -1
    """

    if interaction_pairs is None:
        interaction_pairs = [
            ('card1', 'addr1'),
            ('card1', 'P_emaildomain'),
            ('card1', 'R_emaildomain'),
            ('card2', 'addr1'),
            ('addr1', 'P_emaildomain')
        ]

    if fitted_mappings is None:
        fitted_mappings = {}

    print(f"\n{'='*60}")
    print("INTERACTION FEATURES")
    print(f"{'='*60}")

    for col1, col2 in interaction_pairs:

        if col1 not in data.columns:
            continue

        if col2 not in data.columns:
            continue

        feature_name = f'{col1}_{col2}_combined'

        left = data[col1].fillna(-9999).astype(str)
        right = data[col2].fillna(-9999).astype(str)

        combined = left + '_' + right

        # ==============================================================
        # TRAIN MODE
        # ==============================================================

        if feature_name not in fitted_mappings:

            categories = pd.Series(
                combined.unique()
            ).reset_index(drop=True)

            mapping = {
                val: idx
                for idx, val in enumerate(categories)
            }

            fitted_mappings[feature_name] = mapping

        mapping = fitted_mappings[feature_name]

        data[feature_name] = (
            combined
            .map(mapping)
            .fillna(-1)
            .astype(np.int32)
        )

        print(
            f"Created: {feature_name} "
            f"({len(mapping):,} train categories)"
        )

        del combined
        del left
        del right

        gc.collect()

    return data, fitted_mappings


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def apply_feature_engineering_pipeline(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply complete feature engineering pipeline.
    """

    if config is None:
        config = {
            'entity_cols': ['card1', 'card2', 'addr1'],
            'agg_cols': ['TransactionAmt'],
            'time_col': 'TransactionDT',
            'windows': [1, 24, 168],
            'interaction_pairs': [
                ('card1', 'addr1'),
                ('card1', 'P_emaildomain'),
                ('card2', 'addr1')
            ]
        }

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING PIPELINE")
    print(f"{'='*60}")

    original_feature_count = len(train_data.columns)

    # ==============================================================
    # TIME FEATURES
    # ==============================================================

    train_data = create_time_features(
        train_data,
        config['time_col']
    )

    val_data = create_time_features(
        val_data,
        config['time_col']
    )

    test_data = create_time_features(
        test_data,
        config['time_col']
    )

    # ==============================================================
    # AMOUNT FEATURES
    # ==============================================================

    train_data = create_amount_features(train_data)
    val_data = create_amount_features(val_data)
    test_data = create_amount_features(test_data)

    gc.collect()

    # ==============================================================
    # AGG FEATURES
    # ==============================================================

    train_data, val_data, test_data = create_aggregation_features(
        train_data,
        val_data,
        test_data,
        entity_cols=config['entity_cols'],
        agg_cols=config['agg_cols'],
        time_col=config['time_col'],
        windows=config['windows']
    )

    gc.collect()

    # ==============================================================
    # DEVIATION FEATURES
    # ==============================================================

    train_data, val_data, test_data = create_deviation_features(
        train_data,
        val_data,
        test_data,
        entity_cols=config['entity_cols'],
        agg_cols=config['agg_cols']
    )

    gc.collect()

    # ==============================================================
    # INTERACTION FEATURES
    # ==============================================================

    interaction_mappings = {}

    train_data, interaction_mappings = create_interaction_features(
        train_data,
        config['interaction_pairs'],
        interaction_mappings
    )

    val_data, _ = create_interaction_features(
        val_data,
        config['interaction_pairs'],
        interaction_mappings
    )

    test_data, _ = create_interaction_features(
        test_data,
        config['interaction_pairs'],
        interaction_mappings
    )

    gc.collect()

    final_feature_count = len(train_data.columns)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Original features: {original_feature_count}")
    print(f"Final features: {final_feature_count}")
    print(f"New features added: {final_feature_count - original_feature_count}")

    return train_data, val_data, test_data
