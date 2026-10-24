"""Loading the Kaggle Playground S6E4 data (Predicting Irrigation Need).

The competition data can't be redistributed, so it isn't in the repo. Put
train.csv in ./data, point IRRIGATE_DATA at a folder that has it, or run on
Kaggle with the competition attached.
"""

import os
from pathlib import Path

import pandas as pd

TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]

NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity", "Temperature_C",
    "Humidity", "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh", "Field_Area_hectare",
    "Previous_Irrigation_mm",
]
CATEGORICAL = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type",
    "Water_Source", "Mulching_Used", "Region",
]

_SEARCH = [
    Path(os.environ.get("IRRIGATE_DATA", "data")),
    Path(__file__).resolve().parents[2] / "data",
    Path("/kaggle/input/competitions/playground-series-s6e4"),
    Path("/kaggle/input/playground-series-s6e4"),
    Path.home() / ".mldata" / "s6e4",
]


def data_dir() -> Path:
    for folder in _SEARCH:
        if (folder / "train.csv").exists():
            return folder
    raise FileNotFoundError(
        "train.csv not found. Download it with "
        "`kaggle competitions download -c playground-series-s6e4` into ./data"
    )


def load(split: str = "train") -> pd.DataFrame:
    df = pd.read_csv(data_dir() / f"{split}.csv")
    if TARGET in df:
        df[TARGET] = pd.Categorical(df[TARGET], categories=CLASSES, ordered=True)
    return df


def sample(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """A stratified random sample, small enough for plots and slow algorithms."""
    df = load()
    frac = n / len(df)
    return (
        df.groupby(TARGET, observed=True, group_keys=False)
        .sample(frac=frac, random_state=seed)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def xy(df: pd.DataFrame, numeric_only: bool = False):
    """Features as a one-hot encoded float matrix and the target as class indices."""
    cols = NUMERIC if numeric_only else NUMERIC + CATEGORICAL
    X = pd.get_dummies(df[cols], columns=[] if numeric_only else CATEGORICAL, dtype=float)
    y = df[TARGET].cat.codes.to_numpy()
    return X, y


# ---------- the second dataset: crop yields (used for regression) ----------

_YIELD_SEARCH = [
    Path(os.environ.get("IRRIGATE_YIELD_DATA", "data/yield")),
    Path(__file__).resolve().parents[2] / "data" / "yield",
    Path("/kaggle/input/crop-yield-prediction-dataset"),
    Path.home() / ".mldata" / "yield",
]


def yields() -> pd.DataFrame:
    """Kaggle 'Crop Yield Prediction Dataset' (FAO / World Bank figures, 1990-2013).

    The source file repeats a country-year once per weather station, so rows are
    averaged per country, crop and year. Yield is converted from hg/ha to t/ha."""
    folder = next((f for f in _YIELD_SEARCH if (f / "yield_df.csv").exists()), None)
    if folder is None:
        raise FileNotFoundError("yield_df.csv not found. Download it with "
                                "`kaggle datasets download -d patelris/crop-yield-prediction-dataset` into ./data/yield")
    raw = pd.read_csv(folder / "yield_df.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    out = (raw.groupby(["Area", "Item", "Year"], as_index=False)
              .agg(yield_hg_ha=("hg/ha_yield", "mean"), rain_mm=("average_rain_fall_mm_per_year", "mean"),
                   pesticides_t=("pesticides_tonnes", "mean"), temp_c=("avg_temp", "mean")))
    out["yield_t_ha"] = out.pop("yield_hg_ha") / 10_000
    return out.rename(columns={"Area": "country", "Item": "crop", "Year": "year"})
