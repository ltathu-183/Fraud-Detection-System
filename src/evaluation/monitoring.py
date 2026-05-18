"""
Monitoring & Drift Detection for Fraud Detection
=================================================

Following IEEE-CIS blueprint: STEP 20-21

CRITICAL IN PRODUCTION

Monitors for:
- Data Drift: P_train(x) ≠ P_online(x)
- Concept Drift: P(y|x) changes over time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Monitor and detect data/concept drift in production."""
    
    def __init__(self):
        self.reference_stats = None
        
    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        STEP 20: Data Drift monitoring
        
        PSI measures distribution shift of a feature over time.
        
        Formula: PSI = Σ (current% - reference%) * ln(current% / reference%)
        
        Interpretation:
        - PSI < 0.10: No significant population change
        - 0.10 ≤ PSI < 0.25: Small population change
        - PSI ≥ 0.25: Large population change (investigate)
        
        Args:
            reference: Reference distribution (training data)
            current: Current distribution (online data)
            n_bins: Number of bins for histogram
        
        Returns:
            PSI value
        """
        # Remove NaNs
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        # Calculate breakpoints from reference
        breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Get proportions in each bin
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        curr_counts = np.histogram(current, bins=breakpoints)[0]
        
        ref_props = (ref_counts + 1e-10) / (len(reference) + 1e-10)
        curr_props = (curr_counts + 1e-10) / (len(current) + 1e-10)
        
        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        return float(psi)
    
    def detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        psi_threshold: float = 0.25
    ) -> Dict[str, float]:
        """
        Detect drift in all numerical features.
        
        Args:
            reference_df: Reference data (training)
            current_df: Current data (online)
            psi_threshold: PSI threshold for alert
        
        Returns:
            Dictionary with PSI for each feature
        """
        logger.info("\n" + "="*70)
        logger.info("FEATURE DRIFT DETECTION (Data Drift)")
        logger.info("="*70)
        
        psi_scores = {}
        drifted_features = []
        
        numerical_cols = reference_df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col not in current_df.columns:
                continue
            
            psi = self.calculate_psi(
                reference_df[col].values,
                current_df[col].values
            )
            
            psi_scores[col] = psi
            
            if psi >= psi_threshold:
                drifted_features.append((col, psi))
        
        # Log results
        if drifted_features:
            logger.warning(f"⚠️  DRIFT DETECTED in {len(drifted_features)} features:")
            for feat, psi in sorted(drifted_features, key=lambda x: x[1], reverse=True):
                logger.warning(f"   {feat}: PSI={psi:.4f}")
        else:
            logger.info(f"✓ No significant drift detected")
        
        logger.info("="*70)
        
        return psi_scores
    
    def monitor_model_performance(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        window_size: int = 1000
    ) -> Dict[str, float]:
        """
        Monitor model performance over time.
        
        STEP 20: Concept Drift monitoring
        P(y|x) changes over time.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            window_size: Size of rolling window
        
        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        logger.info("\n" + "="*70)
        logger.info("MODEL PERFORMANCE MONITORING (Concept Drift)")
        logger.info("="*70)
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
        }
        
        # Rolling window evaluation
        n_windows = len(y_true) // window_size
        if n_windows > 1:
            rolling_pr_aucs = []
            
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                
                pr_auc = average_precision_score(
                    y_true[start:end],
                    y_pred_proba[start:end]
                )
                rolling_pr_aucs.append(pr_auc)
            
            metrics['pr_auc_trend'] = rolling_pr_aucs
            metrics['pr_auc_mean'] = np.mean(rolling_pr_aucs)
            metrics['pr_auc_std'] = np.std(rolling_pr_aucs)
        
        logger.info(f"Overall ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Overall PR-AUC:  {metrics['pr_auc']:.4f}")
        
        logger.info("="*70)
        
        return metrics
    
    def monitor_business_metrics(
        self,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Monitor business impact metrics.
        
        Args:
            y_pred_proba: Predicted probabilities
            threshold: Decision threshold
        
        Returns:
            Dictionary with business metrics
        """
        logger.info("\n" + "="*70)
        logger.info("BUSINESS METRICS MONITORING")
        logger.info("="*70)
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        total = len(y_pred)
        approved = (y_pred == 0).sum()
        declined = (y_pred == 1).sum()
        
        metrics = {
            'total_transactions': total,
            'approval_rate': approved / total,
            'decline_rate': declined / total,
            'avg_fraud_score': np.mean(y_pred_proba),
            'max_fraud_score': np.max(y_pred_proba),
            'min_fraud_score': np.min(y_pred_proba),
        }
        
        logger.info(f"Total transactions: {total:,}")
        logger.info(f"Approval rate: {metrics['approval_rate']:.2%}")
        logger.info(f"Decline rate: {metrics['decline_rate']:.2%}")
        logger.info(f"Avg fraud score: {metrics['avg_fraud_score']:.4f}")
        logger.info("="*70)
        
        return metrics
    
    def create_monitoring_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Create comprehensive monitoring report.
        
        Args:
            reference_df: Reference training data
            current_df: Current online data
            y_true: True labels (if available)
            y_pred_proba: Predicted probabilities
        
        Returns:
            Comprehensive monitoring report
        """
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE MONITORING REPORT")
        logger.info("="*70)
        
        report = {
            'data_drift': self.detect_feature_drift(reference_df, current_df),
            'business_metrics': self.monitor_business_metrics(y_pred_proba),
        }
        
        if y_true is not None:
            report['model_performance'] = self.monitor_model_performance(
                y_true, y_pred_proba
            )
        
        logger.info("="*70)
        
        return report
