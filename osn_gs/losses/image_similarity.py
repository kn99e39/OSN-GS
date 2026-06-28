from __future__ import annotations

import numpy as np


def l1_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(np.asarray(prediction) - np.asarray(target)).mean())


def mse_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(prediction) - np.asarray(target)
    return float(np.square(diff).mean())
