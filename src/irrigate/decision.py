"""Turning class probabilities into decisions that maximise balanced accuracy.

argmax(p) maximises plain accuracy. Balanced accuracy weighs every class
equally, so the rare High class should win more often than argmax lets it.
Multiplying each class's probability by a weight and then taking the argmax
moves the decision boundaries; the weights are searched on out-of-fold
predictions so they are never tuned on the data they are scored on.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def decide(proba, weights):
    return np.argmax(proba * np.asarray(weights), axis=1)


def prior_weights(y):
    """The classic correction: divide by the class frequency."""
    freq = np.bincount(y) / len(y)
    return 1 / freq / (1 / freq).sum()


def search_weights(proba, y, grid=None, rounds=3):
    """Coordinate search over per-class weights (the first class stays at 1)."""
    grid = np.geomspace(0.2, 30, 80) if grid is None else grid
    w = np.ones(proba.shape[1])
    best = balanced_accuracy_score(y, decide(proba, w))
    for _ in range(rounds):
        for k in range(1, proba.shape[1]):
            for g in grid:
                trial = w.copy()
                trial[k] = g
                score = balanced_accuracy_score(y, decide(proba, trial))
                if score > best:
                    best, w = score, trial
    return w, best
