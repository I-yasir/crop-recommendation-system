"""Evaluation: performance metrics and cost measurement, as defined in Phase 3."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, top_k_accuracy_score)


def performance_metrics(y_true: np.ndarray, proba: np.ndarray, n_classes: int, k: int = 3) -> dict[str, float]:
    """Accuracy, macro precision/recall/F1, top-k accuracy and log loss from predicted probabilities."""
    labels = np.arange(n_classes)
    pred = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        f"top{k}_accuracy": float(top_k_accuracy_score(y_true, proba, k=k, labels=labels)),
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
    }


def confusion(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> list[list[int]]:
    """Confusion matrix as nested lists (JSON friendly)."""
    return confusion_matrix(y_true, proba.argmax(axis=1), labels=np.arange(n_classes)).tolist()


def median_latency_ms(fn: Callable[[], object], runs: int = 200, warmup: int = 10) -> float:
    """Median wall-clock time of ``fn()`` in milliseconds after a warm-up."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def cost_metrics(model, model_path: Path, sample: pd.DataFrame, batch_size: int = 1000) -> dict[str, float]:
    """Inference latency, model size and in-process memory for a fitted pipeline.

    Memory here is the ``tracemalloc`` peak (Python/NumPy allocations) for reloading the model and
    predicting. The Phase 3 notebook also measures process RSS in a fresh interpreter. This lighter
    version is meant for routine retraining logs.
    """
    batch = sample.sample(batch_size, replace=True, random_state=0)
    one_row = sample.iloc[[0]]

    tracemalloc.start()
    reloaded = joblib.load(model_path)
    reloaded.predict_proba(batch)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "latency_single_ms": median_latency_ms(lambda: model.predict_proba(one_row)),
        "latency_batch1k_ms": median_latency_ms(lambda: model.predict_proba(batch), runs=20, warmup=3),
        "size_kb": model_path.stat().st_size / 1024,
        "py_alloc_peak_mb": peak / 2**20,
    }
