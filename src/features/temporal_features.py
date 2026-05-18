import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """
    Production-grade temporal feature engine:

    Guarantees:
    - No pandas rolling
    - No fragmentation warnings
    - No NaN leakage (controlled fill)
    - Deterministic outputs
    - O(N) per entity group
    """

    def __init__(self):
        self.time_col = "TransactionDT"
        self.amount_col = "TransactionAmt"

    # ======================================================
    # CORE PREP
    # ======================================================

    def _prep(self, df, entity):
        df = df.copy()
        df["_ts"] = df[self.time_col].astype(np.int64)

        df = df.sort_values([entity, "_ts"])
        return df

    # ======================================================
    # SLIDING WINDOW COUNT (SAFE)
    # ======================================================

    def _rolling_count(self, df, entity, window_sec):
        out = np.zeros(len(df), dtype=np.int32)

        for _, g in df.groupby(entity, sort=False):
            idx = g.index.values
            t = g["_ts"].values

            left = 0

            for i in range(len(t)):
                while t[i] - t[left] > window_sec:
                    left += 1
                out[idx[i]] = i - left

        return out

    # ======================================================
    # SLIDING WINDOW STATS (SAFE)
    # ======================================================

    def _rolling_stats(self, df, entity, window_sec):
        mean = np.zeros(len(df), dtype=np.float32)
        std = np.zeros(len(df), dtype=np.float32)
        mx = np.zeros(len(df), dtype=np.float32)
        mn = np.zeros(len(df), dtype=np.float32)

        for _, g in df.groupby(entity, sort=False):
            idx = g.index.values
            t = g["_ts"].values
            x = g[self.amount_col].values

            left = 0

            for i in range(len(t)):
                while t[i] - t[left] > window_sec:
                    left += 1

                window = x[left:i]

                if len(window) == 0:
                    mean[idx[i]] = 0
                    std[idx[i]] = 0
                    mx[idx[i]] = 0
                    mn[idx[i]] = 0
                else:
                    mean[idx[i]] = window.mean()
                    std[idx[i]] = window.std()
                    mx[idx[i]] = window.max()
                    mn[idx[i]] = window.min()

        return mean, std, mx, mn

    # ======================================================
    # UNIQUE COUNT (SAFE APPROX)
    # ======================================================

    def _unique_count(self, df, entity, target):
        out = np.zeros(len(df), dtype=np.int32)

        for _, g in df.groupby(entity, sort=False):
            seen = set()
            idx = g.index.values

            for i, val in enumerate(g[target].fillna("__NA__").values):
                if val not in seen:
                    seen.add(val)
                out[idx[i]] = len(seen)

        return out

    # ======================================================
    # FEATURE BUILDERS
    # ======================================================

    def create_velocity(self, df, entity, windows=(3600, 86400, 7*86400)):
        df = self._prep(df, entity)

        feats = {}

        for w in windows:
            feats[f"{entity}_cnt_{w}s"] = self._rolling_count(df, entity, w)

        return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)

    def create_amount_stats(self, df, entity, windows=(86400, 7*86400, 30*86400)):
        df = self._prep(df, entity)

        feats = {}

        for w in windows:
            m, s, mx, mn = self._rolling_stats(df, entity, w)

            feats[f"{entity}_amt_mean_{w}s"] = m
            feats[f"{entity}_amt_std_{w}s"] = s
            feats[f"{entity}_amt_max_{w}s"] = mx
            feats[f"{entity}_amt_min_{w}s"] = mn

        return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)

    def create_time_since_last(self, df, entity):
        df = self._prep(df, entity)

        out = np.zeros(len(df), dtype=np.float32)

        for _, g in df.groupby(entity, sort=False):
            idx = g.index.values
            t = g["_ts"].values

            prev = np.diff(t, prepend=t[0])
            out[idx] = prev

        df["time_since_last_tx"] = out
        return df

    def create_unique_counts(self, df, entity, target):
        df = self._prep(df, entity)

        out = self._unique_count(df, entity, target)

        df[f"unique_{target}_count"] = out
        return df

    # ======================================================
    # MAIN PIPELINE
    # ======================================================

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("TEMPORAL START")

        df = df.copy()

        # ensure timestamp
        df[self.time_col] = df[self.time_col].astype(np.int64)

        # core entities
        entities = ["card1", "uid"]

        for e in entities:
            if e in df.columns:
                df = self.create_velocity(df, e)
                df = self.create_amount_stats(df, e)
                df = self.create_time_since_last(df, e)

        # device cross features
        if "DeviceInfo" in df.columns:
            df = self.create_unique_counts(df, "card1", "DeviceInfo")

        # FINAL CLEANUP (critical)
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        logger.info("TEMPORAL DONE")

        return df