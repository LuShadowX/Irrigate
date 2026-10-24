"""Evaluation helpers: one place for metrics, confusion matrices and curves."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, auc, average_precision_score, balanced_accuracy_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

from .data import CLASSES
from .plotting import CLASS_COLORS


def scores(y_true, y_pred, proba=None):
    out = {
        "balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro F1": f1_score(y_true, y_pred, average="macro"),
        "High recall": recall_score(y_true, y_pred, labels=[2], average=None)[0],
        "High precision": precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0],
    }
    if proba is not None:
        out["ROC AUC (macro, OvR)"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    return out


def per_class(y_true, y_pred):
    return pd.DataFrame({
        "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=None),
        "F1": f1_score(y_true, y_pred, average=None),
        "support": np.bincount(y_true, minlength=3),
    }, index=CLASSES)


def confusion(y_true, y_pred, ax=None, normalize="true", title=None):
    ax = ax or plt.gca()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASSES, normalize=normalize,
                                            values_format=".2f" if normalize else "d", cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title or "Confusion matrix (rows = true class)")
    ax.grid(False)
    return ax


def roc_pr(y_true, proba, axes=None, name=""):
    """One-vs-rest ROC and precision-recall curves for each class."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    Y = label_binarize(y_true, classes=[0, 1, 2])
    for k, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(Y[:, k], proba[:, k])
        prec, rec, _ = precision_recall_curve(Y[:, k], proba[:, k])
        axes[0].plot(fpr, tpr, color=CLASS_COLORS[cls], label=f"{cls}  AUC {auc(fpr, tpr):.3f}")
        axes[1].plot(rec, prec, color=CLASS_COLORS[cls], label=f"{cls}  AP {average_precision_score(Y[:, k], proba[:, k]):.3f}")
        axes[1].axhline(Y[:, k].mean(), color=CLASS_COLORS[cls], ls=":", lw=1)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title=f"ROC curves {name}".strip())
    axes[1].set(xlabel="recall", ylabel="precision", title=f"Precision–recall curves {name}".strip())
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=9)
    return axes
