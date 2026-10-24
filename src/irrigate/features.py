"""Feature engineering that is safe to put inside a cross-validation pipeline.

Everything here follows the scikit-learn fit/transform contract: statistics are
learned in fit() from the training fold only, so nothing leaks from the
validation rows into the features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

WATER_STAGES = {"Vegetative", "Flowering"}  # the stages where High need actually occurs


def domain_features(df: pd.DataFrame, rain_cut: float) -> pd.DataFrame:
    """Row-wise features from agronomy. Apart from rain_cut (learned on training data)
    they use one row at a time, so they can't leak."""
    out = pd.DataFrame(index=df.index)
    out["dryness"] = df["Temperature_C"] / df["Soil_Moisture"]
    out["evaporation_drive"] = df["Temperature_C"] * df["Wind_Speed_kmh"]
    out["water_supply"] = df["Rainfall_mm"] / 1000 + df["Previous_Irrigation_mm"] / 100
    out["thirsty_stage"] = df["Crop_Growth_Stage"].isin(WATER_STAGES).astype(int)
    out["no_mulch"] = (df["Mulching_Used"] == "No").astype(int)
    out["exposed_thirsty"] = out["thirsty_stage"] * out["no_mulch"]
    out["low_rain"] = (df["Rainfall_mm"] < rain_cut).astype(int)
    return out


class DomainFeatures(BaseEstimator, TransformerMixin):
    """Adds the domain features to a DataFrame (the low-rain cut-off is learned in fit)."""

    def fit(self, X, y=None):
        self.rain_cut_ = float(X["Rainfall_mm"].quantile(0.1))
        return self

    def transform(self, X):
        return pd.concat([X, domain_features(X, self.rain_cut_)], axis=1)


class GroupStats(BaseEstimator, TransformerMixin):
    """Aggregation features: each field compared with the average field of its group
    (e.g. its crop type), using means learned on the training data only."""

    def __init__(self, groups=("Crop_Type", "Region"), columns=("Soil_Moisture", "Temperature_C")):
        self.groups = groups
        self.columns = columns

    def fit(self, X, y=None):
        self.means_ = {g: X.groupby(g)[list(self.columns)].mean() for g in self.groups}
        self.global_ = X[list(self.columns)].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for g in self.groups:
            means = self.means_[g]
            for c in self.columns:
                group_mean = X[g].map(means[c]).astype(float).fillna(self.global_[c])
                X[f"{c}_vs_{g}"] = X[c] - group_mean
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by how often it appears in the training data."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_ = {c: X[c].value_counts(normalize=True) for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([X[c].map(self.freq_[c]).fillna(0).to_numpy(dtype=float) for c in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.freq_])
