"""
Data validation and time-based splitting for fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def time_based_split(
    train_transaction: pd.DataFrame,
    train_identity: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    time_col: str = 'TransactionDT'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data based on time to prevent data leakage.
    
    Args:
        train_transaction: Transaction training data
        train_identity: Identity training data
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        time_col: Column name for time (default: 'TransactionDT')
    
    Returns:
        Tuple of (train_trans, train_id, val_trans, val_id, test_trans, test_id)
    """
    
    merged_data = train_transaction.merge(
        train_identity,
        on='TransactionID',
        how='left'
    )
    
    merged_data = merged_data.sort_values(time_col).reset_index(drop=True)
    
    n_samples = len(merged_data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = merged_data.iloc[:train_end].copy()
    val_data = merged_data.iloc[train_end:val_end].copy()
    test_data = merged_data.iloc[val_end:].copy()
    
    trans_cols = train_transaction.columns.tolist()
    id_cols = train_identity.columns.tolist()
    id_cols.remove('TransactionID')
    
    train_trans = train_data[trans_cols].copy()
    val_trans = val_data[trans_cols].copy()
    test_trans = test_data[trans_cols].copy()
    
    train_id = train_data[['TransactionID'] + id_cols].copy()
    val_id = val_data[['TransactionID'] + id_cols].copy()
    test_id = test_data[['TransactionID'] + id_cols].copy()
    
    print(f"\n{'='*60}")
    print("TIME-BASED SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {n_samples}")
    print(f"Train samples: {len(train_trans)} ({len(train_trans)/n_samples:.2%})")
    print(f"Validation samples: {len(val_trans)} ({len(val_trans)/n_samples:.2%})")
    print(f"Test samples: {len(test_trans)} ({len(test_trans)/n_samples:.2%})")
    
    print(f"\nTime ranges ({time_col}):")
    print(f"Train: {train_data[time_col].min():.0f} - {train_data[time_col].max():.0f}")
    print(f"Validation: {val_data[time_col].min():.0f} - {val_data[time_col].max():.0f}")
    print(f"Test: {test_data[time_col].min():.0f} - {test_data[time_col].max():.0f}")
    
    return train_trans, train_id, val_trans, val_id, test_trans, test_id


def print_fraud_distribution(
    train_trans: pd.DataFrame,
    val_trans: pd.DataFrame,
    test_trans: pd.DataFrame,
    label_col: str = 'isFraud'
) -> None:
    """
    Print fraud distribution across splits to check consistency.
    
    Args:
        train_trans: Training transaction data
        val_trans: Validation transaction data
        test_trans: Test transaction data
        label_col: Name of the target column
    """
    
    print(f"\n{'='*60}")
    print("FRAUD DISTRIBUTION ACROSS SPLITS")
    print(f"{'='*60}")
    
    for name, data in [('Train', train_trans), ('Validation', val_trans), ('Test', test_trans)]:
        total = len(data)
        fraud = data[label_col].sum()
        legit = total - fraud
        fraud_rate = fraud / total
        
        print(f"\n{name}:")
        print(f"  Total: {total:,}")
        print(f"  Fraud: {fraud:,} ({fraud_rate:.4%})")
        print(f"  Legitimate: {legit:,} ({1-fraud_rate:.4%})")
