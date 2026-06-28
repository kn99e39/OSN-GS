from __future__ import annotations

from typing import Iterable, Protocol, TypeAlias

import numpy as np

ArrayLike: TypeAlias = Iterable[float] | np.ndarray
Vector3: TypeAlias = np.ndarray
RGB: TypeAlias = np.ndarray


class RenderableGaussianSet(Protocol):
    @property
    def positions(self) -> np.ndarray:
        ...

    @property
    def colors(self) -> np.ndarray:
        ...

    @property
    def opacities(self) -> np.ndarray:
        ...
