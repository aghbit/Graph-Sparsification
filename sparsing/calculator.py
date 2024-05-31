from networkit.linkprediction import (
    JaccardIndex,
    CommonNeighborsIndex,
    PreferentialAttachmentIndex,
    AdamicAdarIndex,
    AdjustedRandIndex,
    AlgebraicDistanceIndex,
    KatzIndex
)

import numpy as np
from networkit.graph import Graph


class Calculate:
    def __init__(self, method, minmax=False, norm=False):
        self._method = method
        self._minmax = minmax
        self._norm = norm

    def run(self, graph: Graph) -> np.ndarray:
        index = self._method(graph)
        scores = [index.run(u, v) for u, v in graph.iterEdges()]
        scores = np.array(scores, dtype=np.float32)
        scores = scores[np.isfinite(scores)]
        if self._minmax:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + np.finfo(np.float32).eps)
        if self._norm:
            scores = (scores - np.mean(scores)) / (np.std(scores) + np.finfo(np.float32).eps)
        return scores


class JaccardCalc(Calculate):
    def __init__(self, method=JaccardIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class CommonNeighborsCalc(Calculate):
    def __init__(self, method=CommonNeighborsIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class PreferentialAttachmentCalc(Calculate):
    def __init__(self, method=PreferentialAttachmentIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class AdamicAdarCalc(Calculate):
    def __init__(self, method=AdamicAdarIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class AdjustedRandCalc(Calculate):
    def __init__(self, method=AdjustedRandIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class AlgebraicDistanceCalc(Calculate):
    def __init__(self, method=AlgebraicDistanceIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)


class KatzCalc(Calculate):
    def __init__(self, method=KatzIndex, minmax=False, norm=True):
        super().__init__(method, minmax=minmax, norm=norm)