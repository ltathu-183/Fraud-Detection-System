"""
Feature Store for Fraud Detection
==================================

Following IEEE-CIS blueprint: STEP 8, 18

Store reusable aggregates for fast online inference.

Features must be:
- fast: low latency
- causal: no future leakage
- cached: scalable
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Central repository for pre-computed features.
    
    Enables fast online inference by pre-calculating:
    - Entity aggregates (card, device, address, email)
    - Transaction statistics
    - Behavioral profiles
    """
    
    def __init__(self):
        """Initialize feature store."""
        self.features = {}  # key -> dict of aggregates
        self.metadata = {}
        
    def build_card_features(
        self,
        df: pd.DataFrame,
        card_col: str = 'card1',
        time_col: str = 'TransactionDT',
        amount_col: str = 'TransactionAmt'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build card-level aggregates.
        
        STEP 8: Feature Store
        
        Pre-computed features:
        - transaction velocity
        - amount statistics
        - fraud rate
        
        Args:
            df: Transaction data (sorted by time)
            card_col: Card column name
            time_col: Time column name
            amount_col: Amount column name
        
        Returns:
            Dictionary of card -> features
        """
        logger.info(f"Building card features from {len(df):,} transactions...")
        
        card_features = {}
        
        for card in df[card_col].unique():
            card_data = df[df[card_col] == card]
            
            card_features[str(card)] = {
                'tx_count': len(card_data),
                'avg_amount': float(card_data[amount_col].mean()),
                'std_amount': float(card_data[amount_col].std() or 0),
                'max_amount': float(card_data[amount_col].max()),
                'min_amount': float(card_data[amount_col].min()),
                'fraud_rate': float((card_data['isFraud'] == 1).mean()),
                'last_tx_time': float(card_data[time_col].max()),
            }
        
        self.features['card'] = card_features
        logger.info(f"  Stored features for {len(card_features):,} unique cards")
        
        return card_features
    
    def build_device_features(
        self,
        df: pd.DataFrame,
        device_col: str = 'DeviceInfo'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build device-level aggregates using vectorized groupby operations.
        
        Args:
            df: Transaction data
            device_col: Device column name
        
        Returns:
            Dictionary of device -> features
        """
        if device_col not in df.columns:
            logger.warning(f"Column '{device_col}' was dropped or not found. Skipping device features.")
            return {}
            
        logger.info(f"Building device features from {len(df):,} transactions...")
        
        # 1. Loại bỏ các dòng có DeviceInfo bị khuyết thiếu (NaN) trước khi gom nhóm
        df_valid = df.dropna(subset=[device_col])
        
        # 2. Sử dụng groupby + agg để tính toán song song tất cả các metrics
        # Cách này tránh việc lọc đi lọc lại DataFrame nhiều lần
        device_stats = df_valid.groupby(device_col).agg(
            tx_count=('TransactionAmt', 'count'),
            fraud_rate=('isFraud', lambda x: float((x == 1).mean())),
            avg_amount=('TransactionAmt', lambda x: float(x.mean()))
        ).to_dict(orient='index')
        
        # 3. Ép kiểu key sang string (để đảm bảo tính nhất quán khi lưu file JSON)
        device_features = {str(device): metrics for device, metrics in device_stats.items()}
        
        # 4. Lưu lại vào bộ nhớ của Feature Store
        self.features['device'] = device_features
        logger.info(f"  Stored features for {len(device_features):,} unique devices")
        
        return device_features

    def build_address_features(
        self,
        df: pd.DataFrame,
        addr_col: str = 'addr1'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build address-level aggregates.
        
        Args:
            df: Transaction data
            addr_col: Address column name
        
        Returns:
            Dictionary of address -> features
        """
        logger.info(f"Building address features from {len(df):,} transactions...")
        
        addr_features = {}
        
        for addr in df[addr_col].dropna().unique():
            addr_data = df[df[addr_col] == addr]
            
            addr_features[str(addr)] = {
                'tx_count': len(addr_data),
                'fraud_rate': float((addr_data['isFraud'] == 1).mean()),
            }
        
        self.features['address'] = addr_features
        logger.info(f"  Stored features for {len(addr_features):,} unique addresses")
        
        return addr_features
    
    def get_card_features(self, card: str) -> Dict[str, Any]:
        """
        Fetch pre-computed card features for online inference.
        
        STEP 18: Online Inference Pipeline
        
        Must be: fast, causal, cached
        
        Args:
            card: Card identifier
        
        Returns:
            Dictionary of features (or empty dict if not found)
        """
        if 'card' not in self.features:
            return {}
        
        return self.features['card'].get(str(card), {})
    
    def get_device_features(self, device: str) -> Dict[str, Any]:
        """
        Fetch pre-computed device features for online inference.
        
        Args:
            device: Device identifier
        
        Returns:
            Dictionary of features
        """
        if 'device' not in self.features:
            return {}
        
        return self.features['device'].get(str(device), {})
    
    def save_to_json(self, filepath: str):
        """
        Save feature store to JSON (for production caching).
        
        Args:
            filepath: Path to save JSON file
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.features, f, indent=2)
        logger.info(f"Feature store saved to {filepath}")
    
    def load_from_json(self, filepath: str):
        """
        Load feature store from JSON (for production serving).
        
        Args:
            filepath: Path to load JSON file
        """
        import json
        with open(filepath, 'r') as f:
            self.features = json.load(f)
        logger.info(f"Feature store loaded from {filepath}")
    
    def build_full_store(
        self,
        df: pd.DataFrame
    ):
        """
        Build complete feature store from training data.
        
        Args:
            df: Training transaction data (sorted by time)
        """
        logger.info("\n" + "="*70)
        logger.info("BUILDING FEATURE STORE")
        logger.info("="*70)
        
        self.build_card_features(df)
        self.build_device_features(df)
        self.build_address_features(df)
        
        logger.info("="*70 + "\n")
