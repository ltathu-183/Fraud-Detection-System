"""
Tiered Fraud Detection Pipeline
================================
Orchestrates the multi-tier fraud detection system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import time

from src.rules.rules_engine import RulesEngine, create_rules_engine
from src.models.lightgbm_model import LightGBMFraudModel, create_lightgbm_model
from src.evaluation.metrics import calculate_metrics, optimize_threshold


class TieredFraudPipeline:
    """
    Multi-tier fraud detection pipeline.
    
    Architecture:
    - Tier 1: Rules Engine (<5ms, filters ~90%)
    - Tier 2: LightGBM Model (<50ms, handles remaining ~10%)
    - Tier 3: Optional Deep Learning (for complex cases only)
    """
    
    def __init__(
        self,
        use_tier_1: bool = True,
        use_tier_2: bool = True,
        use_tier_3: bool = False
    ):
        """
        Initialize the tiered pipeline.
        
        Args:
            use_tier_1: Enable Tier 1 (Rules Engine)
            use_tier_2: Enable Tier 2 (LightGBM)
            use_tier_3: Enable Tier 3 (Deep Learning - requires GPU)
        """
        self.use_tier_1 = use_tier_1
        self.use_tier_2 = use_tier_2
        self.use_tier_3 = use_tier_3
        
        self.tier_1_engine = None
        self.tier_2_model = None
        self.tier_3_model = None
        
        self.tier_1_threshold = 0.0
        self.tier_2_threshold = 0.5
        
        if self.use_tier_1:
            self.tier_1_engine = create_rules_engine()
            print("Tier 1 (Rules Engine) initialized")
        
        if self.use_tier_2:
            self.tier_2_model = create_lightgbm_model()
            print("Tier 2 (LightGBM) initialized")
        
        if self.use_tier_3:
            print("Tier 3 (Deep Learning) not implemented - requires GPU")
            self.use_tier_3 = False
    
    def train_tier_2(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list = None
    ):
        """
        Train Tier 2 LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
        """
        if not self.use_tier_2:
            print("Tier 2 is not enabled")
            return
        
        print("\n" + "="*60)
        print("TRAINING TIER 2 MODEL")
        print("="*60)
        
        self.tier_2_model.train(X_train, y_train, X_val, y_val, feature_names)
        
        # Optimize threshold
        val_predictions = self.tier_2_model.predict(X_val)
        self.tier_2_threshold, _ = optimize_threshold(
            y_val, val_predictions, metric='f_beta', beta=2
        )
        
        print(f"\nOptimal Tier 2 threshold: {self.tier_2_threshold:.4f}")
    
    def predict_single(self, transaction: pd.Series) -> Dict:
        """
        Predict fraud for a single transaction through the tiered pipeline.
        
        Args:
            transaction: Transaction data as pandas Series
        
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        result = {
            'transaction_id': transaction.get('TransactionID'),
            'final_prediction': None,
            'final_probability': None,
            'tier_used': None,
            'tier_1_result': None,
            'tier_2_result': None,
            'latency_ms': None
        }
        
        # Tier 1: Rules Engine
        if self.use_tier_1:
            tier_1_start = time.time()
            is_flagged, rule_results = self.tier_1_engine.evaluate_transaction(transaction)
            tier_1_latency = (time.time() - tier_1_start) * 1000
            
            result['tier_1_result'] = {
                'flagged': is_flagged,
                'confidence': max([r.confidence for r in rule_results]),
                'reason': [r.reason for r in rule_results if r.is_flagged],
                'latency_ms': tier_1_latency
            }
            
            # If Tier 1 flags as safe, return immediately
            if not is_flagged:
                result['final_prediction'] = 0
                result['final_probability'] = 0.0
                result['tier_used'] = 'tier_1'
                result['latency_ms'] = (time.time() - start_time) * 1000
                return result
        
        # Tier 2: LightGBM Model
        if self.use_tier_2:
            tier_2_start = time.time()
            
            # Convert transaction to feature array
            if self.tier_2_model.feature_names:
                feature_array = transaction[self.tier_2_model.feature_names].values.reshape(1, -1)
            else:
                feature_array = transaction.values.reshape(1, -1)
            
            probability = self.tier_2_model.predict(feature_array)[0]
            prediction = 1 if probability >= self.tier_2_threshold else 0
            
            tier_2_latency = (time.time() - tier_2_start) * 1000
            
            result['tier_2_result'] = {
                'probability': probability,
                'prediction': prediction,
                'threshold': self.tier_2_threshold,
                'latency_ms': tier_2_latency
            }
            
            result['final_prediction'] = prediction
            result['final_probability'] = probability
            result['tier_used'] = 'tier_2'
        
        # Tier 3: Deep Learning (optional, not implemented)
        if self.use_tier_3:
            pass
        
        result['latency_ms'] = (time.time() - start_time) * 1000
        
        return result
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for a batch of transactions.
        
        Args:
            data: DataFrame of transactions
        
        Returns:
            DataFrame with prediction results
        """
        results = []
        
        for idx, row in data.iterrows():
            result = self.predict_single(row)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def evaluate_pipeline(
        self,
        data: pd.DataFrame,
        y_true: np.ndarray,
        label_col: str = 'isFraud'
    ) -> Dict:
        """
        Evaluate the tiered pipeline performance.
        
        Args:
            data: Transaction data
            y_true: True labels
            label_col: Label column name
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict_batch(data)
        
        y_pred_proba = predictions['final_probability'].values
        y_pred = predictions['final_prediction'].values
        
        metrics = calculate_metrics(y_true, y_pred_proba, self.tier_2_threshold)
        
        # Add tier-specific statistics
        tier_usage = predictions['tier_used'].value_counts()
        
        metrics['tier_1_usage'] = tier_usage.get('tier_1', 0)
        metrics['tier_2_usage'] = tier_usage.get('tier_2', 0)
        metrics['tier_1_percentage'] = metrics['tier_1_usage'] / len(predictions) if len(predictions) > 0 else 0
        metrics['tier_2_percentage'] = metrics['tier_2_usage'] / len(predictions) if len(predictions) > 0 else 0
        
        # Average latency
        metrics['avg_latency_ms'] = predictions['latency_ms'].mean()
        
        print("\n" + "="*60)
        print("PIPELINE EVALUATION")
        print("="*60)
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"\nTier Usage:")
        print(f"  Tier 1: {metrics['tier_1_usage']:,} ({metrics['tier_1_percentage']:.1%})")
        print(f"  Tier 2: {metrics['tier_2_usage']:,} ({metrics['tier_2_percentage']:.1%})")
        print(f"\nAverage Latency: {metrics['avg_latency_ms']:.2f}ms")
        print("="*60)
        
        return metrics
    
    def get_pipeline_stats(self) -> Dict:
        """
        Get pipeline statistics and configuration.
        
        Returns:
            Dictionary with pipeline stats
        """
        stats = {
            'tier_1_enabled': self.use_tier_1,
            'tier_2_enabled': self.use_tier_2,
            'tier_3_enabled': self.use_tier_3,
            'tier_2_threshold': self.tier_2_threshold,
            'tier_1_engine_initialized': self.tier_1_engine is not None,
            'tier_2_model_trained': self.tier_2_model.model is not None if self.tier_2_model else False
        }
        
        return stats


def create_tiered_pipeline(
    use_tier_1: bool = True,
    use_tier_2: bool = True,
    use_tier_3: bool = False
) -> TieredFraudPipeline:
    """
    Factory function to create a tiered pipeline instance.
    
    Args:
        use_tier_1: Enable Tier 1
        use_tier_2: Enable Tier 2
        use_tier_3: Enable Tier 3
    
    Returns:
        Configured TieredFraudPipeline instance
    """
    return TieredFraudPipeline(
        use_tier_1=use_tier_1,
        use_tier_2=use_tier_2,
        use_tier_3=use_tier_3
    )
