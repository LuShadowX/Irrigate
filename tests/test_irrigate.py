import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from irrigate import decision, features, preprocess
from irrigate.data import CATEGORICAL, NUMERIC


def make_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.uniform(1, 50, n) for c in NUMERIC})
    choices = {
        "Soil_Type": ["Clay", "Loamy"], "Crop_Type": ["Rice", "Wheat", "Maize"],
        "Crop_Growth_Stage": preprocess.STAGE_ORDER, "Season": ["Kharif", "Rabi"],
        "Irrigation_Type": ["Drip", "Canal"], "Water_Source": ["River", "Groundwater"],
        "Mulching_Used": ["Yes", "No"], "Region": ["North", "South"],
    }
    for c in CATEGORICAL:
        df[c] = rng.choice(choices[c], n)
    y = (df["Temperature_C"] / df["Soil_Moisture"] > 1.5).astype(int) + (df["Temperature_C"] / df["Soil_Moisture"] > 6).astype(int)
    return df, y.to_numpy()


def test_domain_features_learn_cutoff_on_training_data_only():
    train, _ = make_frame(seed=1)
    test, _ = make_frame(seed=2)
    test["Rainfall_mm"] = 1000.0  # every test field is very wet
    f = features.DomainFeatures().fit(train)
    out = f.transform(test)
    assert f.rain_cut_ == train["Rainfall_mm"].quantile(0.1)
    assert out["low_rain"].sum() == 0  # cut-off from train, not recomputed on test


def test_group_stats_use_training_means():
    train, _ = make_frame(seed=3)
    g = features.GroupStats(groups=("Crop_Type",), columns=("Soil_Moisture",)).fit(train)
    test = train.iloc[:5].copy()
    out = g.transform(test)
    expected = test["Soil_Moisture"] - test["Crop_Type"].map(train.groupby("Crop_Type")["Soil_Moisture"].mean())
    assert np.allclose(out["Soil_Moisture_vs_Crop_Type"], expected)


def test_frequency_encoder():
    enc = features.FrequencyEncoder().fit(pd.DataFrame({"c": ["a", "a", "a", "b"]}))
    assert np.allclose(enc.transform(pd.DataFrame({"c": ["a", "b", "z"]})).ravel(), [0.75, 0.25, 0.0])


def test_pipeline_runs_end_to_end_with_engineered_features():
    df, y = make_frame()
    for pipe in (preprocess.pipeline(LogisticRegression(max_iter=500), engineered=True),
                 preprocess.group_pipeline(LogisticRegression(max_iter=500))):
        pipe.fit(df, y)
        assert pipe.predict(df).shape == (len(df),)


def test_weight_search_beats_argmax_on_imbalanced_probabilities():
    rng = np.random.default_rng(0)
    y = rng.choice(3, 5000, p=[0.6, 0.37, 0.03])
    logits = rng.normal(0, 1, (5000, 3)) + np.eye(3)[y] * 1.2 + np.log([0.6, 0.37, 0.03])
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    base = balanced_accuracy_score(y, proba.argmax(1))
    w, best = decision.search_weights(proba, y)
    assert best > base + 0.05
    assert np.isclose(best, balanced_accuracy_score(y, decision.decide(proba, w)))
    assert w[2] > w[0]  # the rare class gets boosted
