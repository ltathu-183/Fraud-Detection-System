import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureSelectionPipeline:
    """
    Drop-in feature selection layer for fraud pipeline
    """

    def __init__(self):
        self.selected_features = None

    # ==============================
    # CONSTANT FILTER
    # ==============================

    def _constant_filter(self, X, threshold=0.999):
        drop = []

        for col in X.columns:
            top = X[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top >= threshold:
                drop.append(col)

        logger.info(f"[FeatureSelection] constant drop: {len(drop)}")
        return X.drop(columns=drop, errors="ignore")

    # ==============================
    # LOW VARIANCE FILTER
    # ==============================

    def _variance_filter(self, X, threshold=1e-5):
        drop = []

        for col in X.columns:
            if X[col].dtype.kind in "bifc":
                if X[col].var() < threshold:
                    drop.append(col)

        logger.info(f"[FeatureSelection] variance drop: {len(drop)}")
        return X.drop(columns=drop, errors="ignore")

    # ==============================
    # LIGHTWEIGHT IMPORTANCE (FAST)
    # ==============================

    def _lgb_filter(self, X, y, top_k=150):
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X.fillna(0), y)

        imp = pd.Series(model.feature_importances_, index=X.columns)
        selected = imp.sort_values(ascending=False).head(top_k).index

        logger.info(f"[FeatureSelection] LGB keep: {len(selected)}")

        return X[selected]

    # ==============================
    # FIT
    # ==============================

    def fit(self, X, y):
        X = X.copy()

        X = self._constant_filter(X)
        X = self._variance_filter(X)
        X = self._lgb_filter(X, y)

        self.selected_features = X.columns.tolist()

        logger.info(f"[FeatureSelection] FINAL: {len(self.selected_features)} features")

        return self

    # ==============================
    # TRANSFORM
    # ==============================

    def transform(self, X):
        return X[self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)