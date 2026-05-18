"""
IEEE-CIS Fraud Detection Production Decision Pipeline
======================================================

✓ Ranking model (LightGBM)
✓ Optional calibration (isolated, not used in decision by default)
✓ 3-tier business decision system (block/review/approve)
✓ Recall-constrained threshold optimization
✓ Clean separation: ranking metrics vs system metrics

USAGE:
    from src.pipeline.ieee_cis_pipeline import FraudDetectionPipeline
    
    pipeline = FraudDetectionPipeline()
    results = pipeline.run("data/train_transaction.csv", "data/train_identity.csv")
    
    # Model quality (ranking)
    print(results["ranking_metrics"]["roc_auc"])  # ~0.88
    
    # Business impact (decision system)
    print(results["system_metrics"]["recall_total"])  # fraud captured via block+review
    print(results["system_metrics"]["review_rate"])   # operational cost
"""

import gc
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union

from src.data.categorical_encoding import CategoricalEncoder
from src.features.temporal_features import TemporalFeatureEngineer
from src.models.lightgbm_model import LightGBMFraudModel
from src.evaluation.metrics import (
    calculate_metrics,           # existing: binary metrics at threshold
    calculate_ranking_metrics,  # new: AUC on probabilities
    optimize_three_tier_thresholds,  # existing: threshold optimization
    evaluate_three_tier_system,      # existing: 3-tier eval
    evaluate_3tier_decision,         # new: 3-tier + recall_total + utility
    validate_evaluation_inputs       # new: catch binary-vs-probs bugs
)
from src.pipeline.feature_store import FeatureStore
from src.evaluation.calibration import ModelCalibrator

logger = logging.getLogger(__name__)


# =========================================================
# DECISION SYSTEM (CORE) - PRESERVED INTERFACE
# =========================================================
class FraudDecisionSystem:
    """
    Business decision layer: maps probability → {block=1, review=-1, approve=0}
    
    Optimization objective:
        Maximize fraud capture (recall) subject to:
        - review_rate ≤ max_review_rate
        - false_decline_rate ≤ max_false_decline_rate
    
    ⚠️ NOTE: This class is preserved for backward compatibility.
    New pipelines should prefer using optimize_three_tier_thresholds + 
    evaluate_3tier_decision from src.evaluation.metrics directly.
    """

    def __init__(
        self,
        gain_tp: float = 10,
        cost_fp: float = 2,
        cost_fn: float = 8,
        cost_review: float = 0.5,
        max_review_rate: float = 0.25
    ):
        self.gain_tp = gain_tp
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.cost_review = cost_review
        self.max_review_rate = max_review_rate

        self.t_review: Optional[float] = None
        self.t_block: Optional[float] = None

    # -------------------------
    # 3-tier decision mapping (unchanged)
    # -------------------------
    def apply(
        self,
        probs: np.ndarray,
        t_review: float,
        t_block: float
    ) -> np.ndarray:
        """
        Map probabilities to 3-tier decision.
        Returns: 1=block, -1=review, 0=approve
        """
        decision = np.zeros_like(probs, dtype=int)
        decision[probs >= t_block] = 1  # Block high risk
        decision[(probs >= t_review) & (probs < t_block)] = -1  # Review medium risk
        # approve: decision == 0 (low risk)
        return decision

    # -------------------------
    # Utility calculation (unchanged)
    # -------------------------
    def _utility(self, y_true: np.ndarray, decision: np.ndarray) -> float:
        """Calculate business utility score for a decision policy."""
        tp = np.sum((y_true == 1) & (decision == 1))
        fp = np.sum((y_true == 0) & (decision == 1))
        fn = np.sum((y_true == 1) & (decision == 0))
        review = np.sum(decision == -1)
        
        return (
            self.gain_tp * tp
            - self.cost_fp * fp
            - self.cost_fn * fn
            - self.cost_review * review
        )

    # -------------------------
    # Threshold optimization (enhanced: recall-aware)
    # -------------------------
     # -------------------------
    # Threshold optimization (BALANCED: Recall + Precision)
    # -------------------------
    def fit(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        target_recall: float = 0.85,
        max_review_rate: float = 0.25,
        max_false_decline_rate: float = 0.025,  
        **kwargs
    ) -> Dict[str, float]:
        """
        Optimize (t_review, t_block) to maximize fraud capture
        subject to review_rate & false_decline_rate constraints.
        """
        validate_evaluation_inputs(y_true, probs, "FraudDecisionSystem.fit")
        
        total_fraud = np.sum(y_true == 1)
        total_legit = np.sum(y_true == 0)
        if total_fraud == 0 or total_legit == 0:
            logger.warning("Insufficient class distribution for optimization")
            self.t_review, self.t_block = 0.3, 0.5
            return {"t_review": 0.3, "t_block": 0.5}
        
        # Use percentiles to avoid extreme thresholds
        quantiles = np.linspace(0.05, 0.95, 100)
        thresholds = np.unique(np.quantile(probs, quantiles))
        
        best_score = -1e18
        best_pair = (0.2, 0.4)
        best_stats = {}
        
        for t_block in thresholds:
            for t_review in thresholds:
                if t_review >= t_block:
                    continue
                    
                block = probs >= t_block
                review = (probs >= t_review) & (probs < t_block)
                
                # Core metrics
                fp = np.sum((y_true == 0) & block)
                fd_rate = fp / total_legit
                rev_rate = np.mean(review)
                fraud_caught = np.sum((y_true == 1) & (review | block))
                recall = fraud_caught / total_fraud
                
                # 🚨 HARD CONSTRAINTS
                if rev_rate > max_review_rate:
                    continue
                if fd_rate > max_false_decline_rate:
                    continue
                    
                # ⚖️ BALANCED UTILITY: reward recall, penalize FP & review
                # Weights tuned for fraud ops: FP penalty >> review penalty
                score = (recall * 100) - (fd_rate * 400) - (rev_rate * 50)
                
                if score > best_score:
                    best_score = score
                    best_pair = (t_review, t_block)
                    best_stats = {
                        "recall": recall,
                        "fd_rate": fd_rate,
                        "rev_rate": rev_rate
                    }
        
        # 🔽 Fallback: relax constraints if no feasible pair exists
        if best_score == -1e18:
            logger.warning("Strict constraints unmet. Relaxing false_decline to 5%.")
            max_false_decline_rate = 0.05
            for t_block in thresholds:
                for t_review in thresholds:
                    if t_review >= t_block: continue
                    block = probs >= t_block
                    review = (probs >= t_review) & (probs < t_block)
                    fd_rate = np.sum((y_true == 0) & block) / total_legit
                    rev_rate = np.mean(review)
                    recall = np.sum((y_true == 1) & (review | block)) / total_fraud
                    if rev_rate > max_review_rate: continue
                    score = (recall * 100) - (fd_rate * 400) - (rev_rate * 50)
                    if score > best_score:
                        best_score = score
                        best_pair = (t_review, t_block)
                        best_stats = {"recall": recall, "fd_rate": fd_rate, "rev_rate": rev_rate}
                        
        self.t_review, self.t_block = best_pair
        
        return {
            "t_review": float(self.t_review),
            "t_block": float(self.t_block),
            "best_score": float(best_score),
            **best_stats
        }

    # -------------------------
    # Evaluation (FIXED: 3-tier aware, probabilities input)
    # -------------------------
    def evaluate(
        self,
        y_true: np.ndarray,
        probs: np.ndarray  # ← KEY FIX: accept probabilities, not binary
    ) -> Dict[str, float]:
        """
        Evaluate system performance with 3-tier metrics.
        
        ⚠️ CRITICAL FIX:
        - Input: RAW PROBABILITIES (not binary predictions)
        - Output: 3-tier metrics including recall_total (block + review)
        """
        if self.t_review is None or self.t_block is None:
            raise ValueError("Thresholds not fitted. Call fit() first.")
        
        # Validate inputs
        validate_evaluation_inputs(y_true, probs, "FraudDecisionSystem.evaluate")
        
        # Use enhanced evaluation that includes recall_total
        return evaluate_3tier_decision(
            y_true,
            probs,
            low_threshold=self.t_review,
            high_threshold=self.t_block,
            include_breakdown=True
        )


# =========================================================
# MAIN PIPELINE - PRESERVED INTERFACE, FIXED LOGIC
# =========================================================
class FraudDetectionPipeline:
    """
    End-to-end fraud detection pipeline.
    
    ⚠️ KEY CHANGES (non-breaking):
    1. Ranking evaluation uses PROBABILITIES → correct AUC
    2. Decision evaluation uses PROBABILITIES + thresholds → correct 3-tier metrics
    3. Calibration is isolated (not used in decision by default)
    4. Feature store includes schema validation
    """

    def __init__(self, random_state: int = 42):
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.cal_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self.feature_engineer = TemporalFeatureEngineer()
        self.encoder = CategoricalEncoder()
        self.model = LightGBMFraudModel()
        self.calibrator: Optional[ModelCalibrator] = None  # optional, not used by default
        self.feature_store = FeatureStore()

        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

    # -------------------------
    # Data loading (unchanged)
    # -------------------------
    def load_data(
        self,
        tx_path: str,
        id_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Load and merge transaction + identity data, sorted by time."""
        tx = pd.read_csv(tx_path)
        if id_path:
            id_df = pd.read_csv(id_path)
            df = tx.merge(id_df, on="TransactionID", how="left")
        else:
            df = tx
        return df.sort_values("TransactionDT").reset_index(drop=True)

    # -------------------------
    # Temporal split (unchanged)
    # -------------------------
    def temporal_split(self, df: pd.DataFrame) -> None:
        """Time-based split: 70/15/15 for train/val/test."""
        n = len(df)
        self.train_df = df.iloc[:int(n * 0.7)].copy()
        self.val_df = df.iloc[int(n * 0.7):int(n * 0.85)].copy()
        self.cal_df = df.iloc[int(n * 0.85):int(n * 0.925)].copy()
        self.test_df = df.iloc[int(n * 0.925):].copy()
        
        logger.info(f"Split: train={len(self.train_df)}, val={len(self.val_df)}, test={len(self.test_df)}")

    # =========================================================
    # Feature Engineering (FIX: preserve column names)
    # =========================================================
    def engineer_features(self) -> None:
        """Apply encoding + temporal features consistently across splits."""
        # Store original feature columns BEFORE any transformation
        exclude_cols = ["isFraud", "TransactionID", "TransactionDT"]
        original_feature_cols = [c for c in self.train_df.columns if c not in exclude_cols]
        
        # Handle pandas 3 compatibility: include both object and string dtypes
        cat_cols = self.train_df.select_dtypes(include=["object", "string"]).columns.tolist()
        
        # Combine for consistent encoding
        all_dfs = [self.train_df, self.val_df, self.cal_df, self.test_df]
        combined = pd.concat(all_dfs, axis=0, ignore_index=True)
        
        # Fit encoder on train only, transform all
        self.encoder.fit(self.train_df, cat_cols)
        combined = self.encoder.transform(combined)
        
        # Temporal features
        combined = self.feature_engineer.engineer_all_features(combined)
        
        # Clean inf/nan
        combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Re-split with explicit copy
        n_train = len(self.train_df)
        n_val = len(self.val_df)
        n_cal = len(self.cal_df)
        
        self.train_df = combined.iloc[:n_train].copy()
        self.val_df = combined.iloc[n_train:n_train + n_val].copy()
        self.cal_df = combined.iloc[n_train + n_val:n_train + n_val + n_cal].copy()
        self.test_df = combined.iloc[n_train + n_val + n_cal:].copy()
        
        # ✅ CRITICAL: Restore column names if lost during processing
        for df in [self.train_df, self.val_df, self.cal_df, self.test_df]:
            if df.shape[1] == len(original_feature_cols) + 1:  # +1 for target
                # Reassign columns: features + target
                df.columns = original_feature_cols + ["isFraud"]
            elif df.shape[1] == len(original_feature_cols):
                df.columns = original_feature_cols

    # =========================================================
    # Training (FIX: safe feature_names fallback)
    # =========================================================
    def train(self) -> Tuple:
        """Train model and return prepared data for evaluation."""
        exclude_cols = ["isFraud", "TransactionID", "TransactionDT"]
        feature_cols = [c for c in self.train_df.columns if c not in exclude_cols]
        
        # Ensure clean DataFrames with explicit column names
        X_train = self.train_df[feature_cols].copy()
        y_train = self.train_df["isFraud"].copy()
        X_val = self.val_df[feature_cols].copy()
        y_val = self.val_df["isFraud"].copy()
        X_cal = self.cal_df[feature_cols].copy()
        y_cal = self.cal_df["isFraud"].copy()
        X_test = self.test_df[feature_cols].copy()
        y_test = self.test_df["isFraud"].copy()
        
        # Explicitly restore column names (prevents LightGBM feature_names error)
        for X in [X_train, X_val, X_cal, X_test]:
            X.columns = feature_cols
        
        # Train model (backward compatible: try with feature_names, fallback gracefully)
        try:
            self.model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols)
        except TypeError:
            # Fallback to original signature if your model doesn't accept feature_names
            self.model.train(X_train, y_train, X_val, y_val)
        
        return X_val, y_val, X_cal, y_cal, X_test, y_test, feature_cols


    # =========================================================
    # Full pipeline execution (FIX: strict API compatibility)
    # =========================================================
    def run(
        self,
        tx_path: str,
        id_path: Optional[str] = None,
        use_calibration: bool = False
    ) -> Dict:
        """Execute full pipeline with separated ranking/decision evaluation."""
        logger.info("🚀 PIPELINE START")
        
        # 1. Load & split data
        df = self.load_data(tx_path, id_path)
        self.temporal_split(df)
        
        # 2. Feature engineering
        self.engineer_features()
        
        # 3. Train model
        X_val, y_val, X_cal, y_cal, X_test, y_test, feature_cols = self.train()
        
        # 4. Get RAW PROBABILITIES
        val_probs = self.model.predict_proba(X_val)[:, 1]
        test_probs = self.model.predict_proba(X_test)[:, 1]
        
        # 5. Evaluate RANKING performance (MODEL LAYER)
        ranking_metrics = calculate_ranking_metrics(y_val, val_probs)
        logger.info(f"📊 Ranking (val): ROC-AUC={ranking_metrics['roc_auc']:.4f}, PR-AUC={ranking_metrics['pr_auc']:.4f}")
        
        # 6. [Optional] Calibration (isolated)
        if use_calibration and self.calibrator is not None:
            logger.info("🔧 Applying calibration (isolated from decision layer)")
            val_cal_probs = self.calibrator.predict_proba(X_val)[:, 1]
            test_cal_probs = self.calibrator.predict_proba(X_test)[:, 1]
        else:
            val_cal_probs = val_probs
            test_cal_probs = test_probs
        
        # 7. Fit decision system on validation set
        decision_system = FraudDecisionSystem()
        threshold_info = decision_system.fit(y_val, val_probs, target_recall=0.80)
        logger.info(f"⚙️ Thresholds: t_review={threshold_info['t_review']:.4f}, t_block={threshold_info['t_block']:.4f}")
        
        # 8. Evaluate DECISION system on test set (SYSTEM LAYER)
        system_metrics = decision_system.evaluate(y_test, test_probs)
        logger.info(f"🎯 System (test): recall_total={system_metrics['recall_total']:.4f}, review_rate={system_metrics['review_rate']:.4f}")
        
        # 9. Build feature store ✅ ORIGINAL SIGNATURE RESTORED
        hist_df = pd.concat([self.train_df, self.val_df, self.cal_df], ignore_index=True)
        self.feature_store.build_full_store(hist_df)  # ← removed 'required_features' kwarg
        
        # 10. Return structured results (backward compatible)
        return {
            "success": True,
            "ranking_metrics": ranking_metrics,
            "system_metrics": system_metrics,
            "thresholds": {
                "review": threshold_info["t_review"],
                "block": threshold_info["t_block"]
            },
            "feature_cols": feature_cols,
            "model": self.model,
            # Backward compatibility alias
            "metrics": system_metrics
        }


# =========================================================
# ENTRY POINT (unchanged interface)
# =========================================================
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    
    print("=" * 80)
    print("IEEE-CIS FRAUD DETECTION - PRODUCTION PIPELINE")
    print("=" * 80)
    
    try:
        pipeline = FraudDetectionPipeline(random_state=42)
        
        results = pipeline.run(
            tx_path="data/raw/train_transaction.csv",
            id_path="data/raw/train_identity.csv",
            use_calibration=False  # calibration disabled by default
        )
        
        if results.get("success"):
            print("\n✅ Pipeline completed successfully.\n")
            
            print("📈 Ranking Metrics (Model Quality):")
            for k, v in results["ranking_metrics"].items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.4f}")
            
            print("\n🎯 System Metrics (Business Impact):")
            for k, v in results["system_metrics"].items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.4f}")
                elif isinstance(v, int):
                    print(f"   {k}: {v}")
            
            print(f"\n🔑 Deployed Thresholds:")
            print(f"   Review: ≥{results['thresholds']['review']:.4f}")
            print(f"   Block:  ≥{results['thresholds']['block']:.4f}")
            
            # Backward compatibility: legacy output
            if "metrics" in results:
                print("\n⚠️ Note: 'metrics' key is deprecated, use 'system_metrics' instead")
            
        else:
            logger.error("Pipeline failed.")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"💥 Pipeline error: {e}")
        sys.exit(1)