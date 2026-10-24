"""Train the final models on all 630,000 fields and write a Kaggle submission.

    python -m irrigate.submit                         # LightGBM + XGBoost + CatBoost, 5-fold
    python -m irrigate.submit --models lightgbm       # one model, for a quick check
    python -m irrigate.submit --blend-only            # re-blend saved out-of-fold predictions

Each model is trained with 5-fold stratified CV. The out-of-fold probabilities
are used twice: to score the model honestly, and to tune the per-class decision
weights for balanced accuracy (irrigate.decision). The weights are tuned on
folds 1-4's out-of-fold predictions and checked on fold 5 before they are
trusted, so the reported score is not tuned on itself.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from . import data, decision
from .features import DomainFeatures

OUT = Path(__file__).resolve().parents[2] / "submissions"
SEED = 42


def frame(df: pd.DataFrame, domain: DomainFeatures) -> pd.DataFrame:
    """Numeric + engineered columns, categoricals as pandas 'category' (all three libraries read it natively)."""
    X = domain.transform(df[data.NUMERIC + data.CATEGORICAL])
    for c in data.CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def make_model(name: str):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                              random_state=SEED, n_jobs=-1, verbose=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=3000, learning_rate=0.04, max_depth=7, min_child_weight=3, subsample=0.8,
                             colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                             max_cat_to_onehot=8, early_stopping_rounds=150, eval_metric="mlogloss",
                             random_state=SEED, n_jobs=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        # one_hot_max_size: every categorical column here has at most 6 values, so plain one-hot splits are
        # used instead of CatBoost's ordered target statistics, which made CPU training many times slower
        return CatBoostClassifier(iterations=3000, learning_rate=0.06, depth=7, loss_function="MultiClass", one_hot_max_size=10,
                                  od_type="Iter", od_wait=150, random_seed=SEED, verbose=0, thread_count=-1)
    raise ValueError(name)


def fit_one(name, model, X_tr, y_tr, X_va, y_va):
    if name == "lightgbm":
        from lightgbm import early_stopping
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(150, verbose=False)])
        return model, model.best_iteration_
    if name == "xgboost":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return model, model.best_iteration
    cat_idx = [X_tr.columns.get_loc(c) for c in data.CATEGORICAL]
    model.fit(X_tr.astype({c: str for c in data.CATEGORICAL}), y_tr, cat_features=cat_idx,
              eval_set=(X_va.astype({c: str for c in data.CATEGORICAL}), y_va))
    return model, model.get_best_iteration()


def predict(name, model, X):
    if name == "catboost":
        X = X.astype({c: str for c in data.CATEGORICAL})
    return model.predict_proba(X)


def folds_for(y, folds=5):
    fold_of = np.empty(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)):
        fold_of[va] = k
    return fold_of


def run(models, folds=5, blend_only=False):
    train, test = data.load("train"), data.load("test")
    y = train[data.TARGET].cat.codes.to_numpy()
    fold_of = folds_for(y, folds)
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text()) if blend_only and report_path.exists() else {}
    oof, test_proba = {}, {}
    if blend_only:
        for name in models:
            oof[name] = np.load(OUT / f"oof_{name}.npy")
            test_proba[name] = np.load(OUT / f"test_{name}.npy")
            report.setdefault(name, {})["CV balanced accuracy, argmax"] = balanced_accuracy_score(y, oof[name].argmax(1))
        return blend(models, oof, test_proba, y, fold_of, test, report, folds)
    domain = DomainFeatures().fit(train)
    X, X_test = frame(train, domain), frame(test, domain)
    for name in models:
        oof[name] = np.zeros((len(y), 3))
        test_proba[name] = np.zeros((len(X_test), 3))
        t0 = time.perf_counter()
        iters = []
        for k in range(folds):
            tr, va = fold_of != k, fold_of == k
            model, best = fit_one(name, make_model(name), X[tr], y[tr], X[va], y[va])
            oof[name][va] = predict(name, model, X[va])
            test_proba[name] += predict(name, model, X_test) / folds
            iters.append(int(best))
            print(f"  {name} fold {k + 1}: balanced acc (argmax) {balanced_accuracy_score(y[va], oof[name][va].argmax(1)):.5f}, trees {best}", flush=True)
        report[name] = {"seconds": round(time.perf_counter() - t0, 1), "trees per fold": iters,
                        "CV balanced accuracy, argmax": balanced_accuracy_score(y, oof[name].argmax(1))}
        np.save(OUT / f"oof_{name}.npy", oof[name])
        np.save(OUT / f"test_{name}.npy", test_proba[name])

    return blend(models, oof, test_proba, y, fold_of, test, report, folds)


def blend(models, oof, test_proba, y, fold_of, test, report, folds):
    blend_oof = np.mean([oof[m] for m in models], axis=0)
    blend_test = np.mean([test_proba[m] for m in models], axis=0)

    # tune decision weights on folds 0-3, check them on fold 4, then refit on everything
    dev, hold = fold_of != folds - 1, fold_of == folds - 1
    w_dev, _ = decision.search_weights(blend_oof[dev], y[dev])
    w_all, best_all = decision.search_weights(blend_oof, y)
    report["blend"] = {
        "models": list(models),
        "CV balanced accuracy, argmax": balanced_accuracy_score(y, blend_oof.argmax(1)),
        "held-out fold, argmax": balanced_accuracy_score(y[hold], blend_oof[hold].argmax(1)),
        "held-out fold, weights tuned on the other folds": balanced_accuracy_score(y[hold], decision.decide(blend_oof[hold], w_dev)),
        "weights (Low, Medium, High)": [round(float(v), 3) for v in w_all],
        "CV balanced accuracy, tuned weights (in-sample)": best_all,
    }
    labels = np.array(data.CLASSES)[decision.decide(blend_test, w_all)]
    sub = pd.DataFrame({"id": test["id"], data.TARGET: labels})
    sub.to_csv(OUT / "submission.csv", index=False)
    report["submission rows"] = len(sub)
    report["submission class shares"] = sub[data.TARGET].value_counts(normalize=True).round(4).to_dict()
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def baseline():
    """My April 2026 approach, reproduced: one-hot + XGBoost(200 trees) + argmax, scored with the same CV."""
    from xgboost import XGBClassifier
    train = data.load("train")
    y = train[data.TARGET].cat.codes.to_numpy()
    X = pd.get_dummies(train[data.NUMERIC + data.CATEGORICAL], dtype=float)
    oof = np.zeros((len(y), 3))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])
    np.save(OUT / "oof_baseline.npy", oof)
    return balanced_accuracy_score(y, oof.argmax(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["lightgbm", "xgboost", "catboost"], choices=["lightgbm", "xgboost", "catboost"])
    parser.add_argument("--blend-only", action="store_true", help="re-blend the saved out-of-fold predictions")
    parser.add_argument("--baseline", action="store_true", help="also score the original one-hot XGBoost approach")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.baseline:
        score = baseline()
        print(f"baseline (one-hot XGBoost, argmax) CV balanced accuracy: {score:.5f}")
        (OUT / "baseline.json").write_text(json.dumps({"CV balanced accuracy, argmax": score}))
    report = run(args.models, blend_only=args.blend_only)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
