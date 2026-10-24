"""Preprocessing pipelines, built once and reused in every notebook."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC
from .features import DomainFeatures, GroupStats

STAGE_ORDER = ["Sowing", "Vegetative", "Flowering", "Harvest"]
NOMINAL = [c for c in CATEGORICAL if c != "Crop_Growth_Stage"]
ENGINEERED = ["dryness", "evaporation_drive", "water_supply", "thirsty_stage", "no_mulch", "exposed_thirsty", "low_rain"]


def column_transformer(scale="standard", numeric=None, engineered=False):
    """One-hot for nominal columns, ordinal codes for growth stage, scaling for numbers."""
    numeric = list(numeric or NUMERIC) + (ENGINEERED if engineered else [])
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), None: "passthrough"}[scale]
    return ColumnTransformer([
        ("num", scaler, numeric),
        ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL),
        ("stage", OrdinalEncoder(categories=[STAGE_ORDER]), ["Crop_Growth_Stage"]),
    ], verbose_feature_names_out=False)


def pipeline(model, scale="standard", engineered=False):
    """Raw DataFrame in, predictions out. Every learned statistic is fitted on the training fold only."""
    steps = [("features", DomainFeatures())] if engineered else []
    steps += [("columns", column_transformer(scale, engineered=engineered)), ("model", model)]
    return Pipeline(steps)


def group_pipeline(model, scale="standard"):
    """Like pipeline(engineered=True) plus the group-aggregation features."""
    extra = [f"{c}_vs_{g}" for g in ("Crop_Type", "Region") for c in ("Soil_Moisture", "Temperature_C")]
    cols = column_transformer(scale, numeric=NUMERIC + extra, engineered=True)
    return Pipeline([("features", DomainFeatures()), ("groups", GroupStats()), ("columns", cols), ("model", model)])
