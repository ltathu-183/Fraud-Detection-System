"""Point-in-time behavioural features for the supported pipeline.

All history windows use ``[t-W, t)``. Events at the same timestamp are not
visible to one another, making output independent of their input ordering.
No target-derived feature is created.
"""

import numpy as np
import pandas as pd


class TemporalFeatureEngineer:
    def __init__(self, time_col="TransactionDT", amount_col="TransactionAmt"):
        self.time_col = time_col
        self.amount_col = amount_col

    @staticmethod
    def _require_identity(df):
        if "_internal_row_id" not in df:
            raise ValueError("_internal_row_id is required before feature engineering")
        if df["_internal_row_id"].isna().any() or not df["_internal_row_id"].is_unique:
            raise ValueError("Internal row identifiers must be unique and non-null")

    def _entity_features(self, source, entity, windows):
        feature_arrays = {
            f"{entity}_{window}s_count": np.zeros(len(source), dtype=np.int32)
            for window in windows
        }
        feature_arrays.update({
            f"{entity}_{window}s_amount_mean": np.full(len(source), np.nan, dtype=np.float32)
            for window in windows
        })
        time_since_last = np.full(len(source), np.nan, dtype=np.float32)
        row_positions = pd.Series(np.arange(len(source)), index=source.index)
        valid = source[entity].notna()
        ordered = source.loc[valid].sort_values(
            [entity, self.time_col, "_internal_row_id"], kind="mergesort"
        )
        for _, group in ordered.groupby(entity, sort=False, observed=True):
            times = group[self.time_col].to_numpy(dtype=np.int64)
            amounts = group[self.amount_col].to_numpy(dtype=float)
            positions = row_positions.loc[group.index].to_numpy()
            cumulative_amount = np.r_[0.0, np.nan_to_num(amounts, nan=0.0).cumsum()]
            # ``right`` points to the first event at the current timestamp, so
            # every same-time event is excluded from its own history.
            right = np.searchsorted(times, times, side="left")
            previous = right - 1
            has_previous = previous >= 0
            time_since_last[positions[has_previous]] = (
                times[has_previous] - times[previous[has_previous]]
            )
            for window in windows:
                left = np.searchsorted(times, times - window, side="left")
                count = right - left
                feature_arrays[f"{entity}_{window}s_count"][positions] = count
                nonempty = count > 0
                feature_arrays[f"{entity}_{window}s_amount_mean"][positions[nonempty]] = (
                    (cumulative_amount[right[nonempty]] - cumulative_amount[left[nonempty]])
                    / count[nonempty]
                )
        feature_arrays[f"{entity}_time_since_last"] = time_since_last
        return pd.DataFrame(feature_arrays, index=source.index)

    def _novelty_features(self, source, entity):
        """First-seen indicators and entity age, using only events before t."""
        is_new = np.zeros(len(source), dtype=np.int8)
        age = np.full(len(source), np.nan, dtype=np.float32)
        positions = pd.Series(np.arange(len(source)), index=source.index)
        valid = source[entity].notna()
        ordered = source.loc[valid].sort_values(
            [entity, self.time_col, "_internal_row_id"], kind="mergesort"
        )
        for _, group in ordered.groupby(entity, sort=False, observed=True):
            group_positions = positions.loc[group.index].to_numpy()
            times = group[self.time_col].to_numpy(dtype=np.int64)
            first_time = times[0]
            # Same-time peers have no strictly earlier evidence, so they all
            # receive the same first-seen status rather than leaking ordering.
            is_new[group_positions[times == first_time]] = 1
            age[group_positions] = times - first_time
        return pd.DataFrame({
            f"{entity}_is_new": is_new,
            f"{entity}_time_since_first_seen": age,
        }, index=source.index)

    def engineer_all_features(
        self,
        df,
        velocity_entities=("card1",),
        novelty_entities=("card1", "DeviceInfo", "P_emaildomain", "addr1"),
        windows=(3600, 86400, 604800),
    ):
        self._require_identity(df)
        if self.time_col not in df or self.amount_col not in df:
            raise ValueError("TransactionDT and TransactionAmt are required")
        original_ids = df["_internal_row_id"].copy()
        output = df.copy()
        for entity in velocity_entities:
            if entity in output:
                output = output.join(self._entity_features(df, entity, windows))
        for entity in novelty_entities:
            if entity in output:
                output = output.join(self._novelty_features(df, entity))
        output = output.loc[df.index]
        if len(output) != len(df) or not output["_internal_row_id"].equals(original_ids):
            raise AssertionError("Feature engineering changed row identity or order")
        return output.replace([np.inf, -np.inf], np.nan)
