"""
Categorical Encoding for Fraud Detection
==========================================

Following IEEE-CIS blueprint: STEP 6

Avoid leakage: Encoding must be fit ONLY on train set.

Methods:
- Frequency encoding: term frequency
- Count encoding: count of occurrences
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CategoricalEncoder:
    """Encode categorical features without data leakage."""
    
    def __init__(self):
        """Initialize encoder."""
        self.frequency_encodings = {}  # col -> {cat: freq}
        self.count_encodings = {}  # col -> {cat: count}
        
    def fit_frequency_encoding(
        self,
        train_df: pd.DataFrame,
        cat_cols: List[str] = None,
        min_freq: float = 0.01
    ) -> Dict[str, Dict]:
        """
        Fit frequency encoding on training data.
        
        STEP 6: Categorical Encoding
        
        Frequency = count_of_category / total_count
        
        Args:
            train_df: Training data (fit only on this)
            cat_cols: Categorical columns to encode
            min_freq: Minimum frequency threshold
        
        Returns:
            Dictionary of encodings
        """
        logger.info("\n" + "="*70)
        logger.info("FITTING FREQUENCY ENCODING (Train set only)")
        logger.info("="*70)
        
        if cat_cols is None:
            cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        for col in cat_cols:
            if col not in train_df.columns:
                continue
            
            # Calculate frequencies on train set only
            value_counts = train_df[col].value_counts()
            total = value_counts.sum()
            frequencies = (value_counts / total).to_dict()
            
            # Keep encodings for observed categories
            self.frequency_encodings[col] = frequencies
            
            logger.info(f"  {col}: {len(frequencies)} unique values")
        
        logger.info("="*70)
        
        return self.frequency_encodings
    
    def fit_count_encoding(
        self,
        train_df: pd.DataFrame,
        cat_cols: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Fit count encoding on training data.
        
        Count = number of occurrences in training set
        
        Args:
            train_df: Training data (fit only on this)
            cat_cols: Categorical columns to encode
        
        Returns:
            Dictionary of count encodings
        """
        logger.info("\n" + "="*70)
        logger.info("FITTING COUNT ENCODING (Train set only)")
        logger.info("="*70)
        
        if cat_cols is None:
            cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        for col in cat_cols:
            if col not in train_df.columns:
                continue
            
            # Calculate counts on train set only
            value_counts = train_df[col].value_counts().to_dict()
            
            self.count_encodings[col] = value_counts
            
            logger.info(f"  {col}: {len(value_counts)} unique values")
        
        logger.info("="*70)
        
        return self.count_encodings
    
    def apply_frequency_encoding(
        self,
        df: pd.DataFrame,
        cat_cols: List[str] = None,
        unknown_value: float = 0.0
    ) -> pd.DataFrame:
        """
        Apply fitted frequency encoding to new data.
        
        IMPORTANT: Uses encodings fit only on train set.
        Unseen categories get unknown_value.
        
        Args:
            df: Data to encode
            cat_cols: Columns to encode (if None, use fitted columns)
            unknown_value: Value for unseen categories
        
        Returns:
            DataFrame with frequency-encoded columns
        """
        if cat_cols is None:
            cat_cols = list(self.frequency_encodings.keys())
        
        for col in cat_cols:
            if col not in self.frequency_encodings:
                logger.warning(f"Column {col} not in fitted encodings, skipping")
                continue
            
            # Apply encoding, use unknown_value for unseen categories
            df[col + '_freq'] = df[col].map(self.frequency_encodings[col]).fillna(unknown_value)
            
            # Drop original column
            df = df.drop(col, axis=1)
        
        return df
    
    def apply_count_encoding(
        self,
        df: pd.DataFrame,
        cat_cols: List[str] = None,
        unknown_value: int = 0
    ) -> pd.DataFrame:
        """
        Apply fitted count encoding to new data.
        
        Args:
            df: Data to encode
            cat_cols: Columns to encode
            unknown_value: Value for unseen categories
        
        Returns:
            DataFrame with count-encoded columns
        """
        if cat_cols is None:
            cat_cols = list(self.count_encodings.keys())
        
        for col in cat_cols:
            if col not in self.count_encodings:
                logger.warning(f"Column {col} not in fitted encodings, skipping")
                continue
            
            # Apply encoding
            df[col + '_count'] = df[col].map(self.count_encodings[col]).fillna(unknown_value)
            
            # Drop original column
            df = df.drop(col, axis=1)
        
        return df
    
    def fit_transform_frequency(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        cat_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit encoding on train, transform all sets.
        
        CRITICAL: Fit ONLY on train, apply to all.
        
        Args:
            train_df: Training data
            val_df: Validation data (optional)
            test_df: Test data (optional)
            cat_cols: Columns to encode
        
        Returns:
            Tuple of (train_encoded, val_encoded, test_encoded)
        """
        logger.info("\n" + "="*70)
        logger.info("FIT-TRANSFORM FREQUENCY ENCODING")
        logger.info("="*70)
        logger.info("⚠️  FITTING ON TRAIN SET ONLY")
        
        # Fit only on train
        self.fit_frequency_encoding(train_df, cat_cols)
        
        # Transform all sets using train encodings
        train_transformed = self.apply_frequency_encoding(train_df, cat_cols)
        
        val_transformed = None
        if val_df is not None:
            val_transformed = self.apply_frequency_encoding(val_df, cat_cols)
        
        test_transformed = None
        if test_df is not None:
            test_transformed = self.apply_frequency_encoding(test_df, cat_cols)
        
        logger.info("="*70)
        
        return train_transformed, val_transformed, test_transformed
    
    def fit_transform_count(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        cat_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit count encoding on train, transform all sets.
        
        Args:
            train_df: Training data
            val_df: Validation data (optional)
            test_df: Test data (optional)
            cat_cols: Columns to encode
        
        Returns:
            Tuple of (train_encoded, val_encoded, test_encoded)
        """
        logger.info("\n" + "="*70)
        logger.info("FIT-TRANSFORM COUNT ENCODING")
        logger.info("="*70)
        logger.info("⚠️  FITTING ON TRAIN SET ONLY")
        
        # Fit only on train
        self.fit_count_encoding(train_df, cat_cols)
        
        # Transform all sets
        train_transformed = self.apply_count_encoding(train_df, cat_cols)
        
        val_transformed = None
        if val_df is not None:
            val_transformed = self.apply_count_encoding(val_df, cat_cols)
        
        test_transformed = None
        if test_df is not None:
            test_transformed = self.apply_count_encoding(test_df, cat_cols)
        
        logger.info("="*70)
        
        return train_transformed, val_transformed, test_transformed

    def fit(self, train_df: pd.DataFrame, cat_cols: list = None):
        """Fit frequency encoding on train set only."""
        return self.fit_frequency_encoding(train_df, cat_cols)

    def transform(self, df: pd.DataFrame, cat_cols: list = None) -> pd.DataFrame:
        """Apply the fitted frequency encoding to a new dataset."""
        return self.apply_frequency_encoding(df, cat_cols)
