"""
Baseline Models for Fraud Detection
====================================

Following IEEE-CIS blueprint: STEP 10

Create simple baseline models to establish performance floor:
1. Dummy Classifier (always non-fraud)
2. Logistic Regression (linear model)
3. Random Forest (simple ensemble)

These models help understand the data difficulty and imbalance issue.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

logger = logging.getLogger(__name__)


class BaselineModels:
    """Container for baseline fraud detection models."""
    
    def __init__(self):
        self.dummy_model = None
        self.logistic_model = None
        self.rf_model = None
        
    def train_dummy_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Train Dummy Classifier baseline.
        
        BASELINE 1: Always predicts non-fraud.
        
        Purpose: Show imbalance issue.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("\n" + "="*70)
        logger.info("BASELINE 1: DUMMY CLASSIFIER (Always Non-Fraud)")
        logger.info("="*70)
        
        self.dummy_model = DummyClassifier(strategy='most_frequent')
        self.dummy_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.dummy_model.predict(X_val)
        y_pred_proba = self.dummy_model.predict_proba(X_val)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        
        metrics = {
            'model': 'dummy',
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'accuracy': (y_pred == y_val).mean()
        }
        
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"PR-AUC:  {pr_auc:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info("Note: Low PR-AUC indicates severe imbalance")
        logger.info("="*70)
        
        return metrics
    
    def train_logistic_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scale_pos_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Train Logistic Regression baseline.
        
        BASELINE 2: Linear benchmark.
        
        Purpose: Establish linear model baseline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            scale_pos_weight: Class weight for fraud class
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("\n" + "="*70)
        logger.info("BASELINE 2: LOGISTIC REGRESSION (Linear Model)")
        logger.info("="*70)
        
        # Handle class imbalance with class_weight
        class_weight = {
            0: 1.0,
            1: scale_pos_weight
        }
        
        self.logistic_model =LogisticRegression(
            max_iter=200,
            solver='saga',
            n_jobs=-1,
            class_weight='balanced',
            random_state=42
        )

        X_train_clean = np.nan_to_num(X_train, nan=-999).astype(np.float32)
        X_val_clean = np.nan_to_num(X_val, nan=-999).astype(np.float32)

        self.logistic_model.fit(X_train_clean, y_train)
        
        # Predictions
        y_pred_proba = self.logistic_model.predict_proba(X_val_clean)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        
        metrics = {
            'model': 'logistic_regression',
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'n_features': X_train.shape[1]
        }
        
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"PR-AUC:  {pr_auc:.4f}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Class weight (fraud): {scale_pos_weight:.2f}x")
        logger.info("="*70)
        
        return metrics
    
    def train_random_forest_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scale_pos_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Train Random Forest baseline.
        
        BASELINE 3: Basic ensemble baseline.
        
        Purpose: Simple tree-based baseline for comparison.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            scale_pos_weight: Class weight for fraud class
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("\n" + "="*70)
        logger.info("BASELINE 3: RANDOM FOREST (Ensemble Baseline)")
        logger.info("="*70)
        
        # Convert scale_pos_weight to class_weight
        class_weight = {
            0: 1.0,
            1: scale_pos_weight
        }
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        X_train_clean = np.nan_to_num(X_train, nan=-999)
        X_val_clean = np.nan_to_num(X_val, nan=-999)

        self.rf_model.fit(X_train_clean, y_train)
        
        # Predictions
        y_pred_proba = self.rf_model.predict_proba(X_val_clean)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        
        metrics = {
            'model': 'random_forest',
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'n_features': X_train.shape[1]
        }
        
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"PR-AUC:  {pr_auc:.4f}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Class weight (fraud): {scale_pos_weight:.2f}x")
        logger.info("="*70)
        
        return metrics
    
    def train_all_baselines(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> pd.DataFrame:
        """
        Train all baseline models and return comparison.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            DataFrame with baseline comparison
        """
        logger.info("\n" + "="*70)
        logger.info("TRAINING ALL BASELINE MODELS")
        logger.info("="*70)
        
        # Calculate class weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}")
        
        # Train all models
        results = []
        results.append(self.train_dummy_baseline(X_train, y_train, X_val, y_val))
        results.append(self.train_logistic_baseline(
            X_train, y_train, X_val, y_val, scale_pos_weight
        ))
        results.append(self.train_random_forest_baseline(
            X_train, y_train, X_val, y_val, scale_pos_weight
        ))
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        logger.info("\n" + "="*70)
        logger.info("BASELINE COMPARISON")
        logger.info("="*70)
        logger.info(comparison_df.to_string())
        logger.info("="*70)
        
        return comparison_df

# if __name__ == "__main__":

#     import numpy as np
#     from sklearn.datasets import make_classification
#     from sklearn.model_selection import train_test_split

#     print("=" * 80)
#     print("TESTING BASELINE MODELS")
#     print("=" * 80)

#     # Fake fraud dataset
#     X, y = make_classification(
#         n_samples=10000,
#         n_features=50,
#         n_informative=10,
#         n_redundant=5,
#         weights=[0.97, 0.03],   # imbalanced fraud-like
#         random_state=42
#     )

#     X_train, X_val, y_train, y_val = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         stratify=y,
#         random_state=42
#     )

#     model = BaselineModels()

#     results = model.train_all_baselines(
#         X_train,
#         y_train,
#         X_val,
#         y_val
#     )

#     print("\nDONE")
#     print(results)