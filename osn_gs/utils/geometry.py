from __future__ import annotations

import numpy as np


def as_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected points with shape (N, 3).")
    return points


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = as_points(a)
    b = as_points(b)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    return (1.0 - t) * np.asarray(a, dtype=np.float32) + t * np.asarray(b, dtype=np.float32)


def principal_axis(points: np.ndarray) -> np.ndarray:
    points = as_points(points)
    if len(points) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh[0].astype(np.float32)
