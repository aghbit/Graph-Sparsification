from networkit.linkprediction import (
    JaccardIndex,
    CommonNeighborsIndex,
    PreferentialAttachmentIndex,
    AdamicAdarIndex,
    AdjustedRandIndex,
    AlgebraicDistanceIndex,
    KatzIndex
)

from networkit.sparsification import LocalDegreeScore, LocalSimilarityScore, TriangleEdgeScore

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


def get_triangles(G: Graph):
    edge_triangles = TriangleEdgeScore(G)
    edge_triangles.run()
    return edge_triangles.scores()


class JaccardCalc(Calculate):
    def __init__(self, method=JaccardIndex, minmax=False, norm=False):
        super().__init__(method, minmax=minmax, norm=norm)


class CommonNeighborsCalc(Calculate):
    def __init__(self, method=CommonNeighborsIndex, minmax=False, norm=False):
        super().__init__(method, minmax=minmax, norm=norm)


class PreferentialAttachmentCalc(Calculate):
    def __init__(self, method=PreferentialAttachmentIndex, minmax=False, norm=False):
        super().__init__(method, minmax=minmax, norm=norm)


class AdamicAdarCalc(Calculate):
    def __init__(self, method=AdamicAdarIndex, minmax=False, norm=False):
        super().__init__(method, minmax=minmax, norm=norm)


class AdjustedRandCalc(Calculate):
    def __init__(self, method=AdjustedRandIndex, minmax=False, norm=False):
        super().__init__(method, minmax=minmax, norm=norm)


class AlgebraicDistanceCalc(Calculate):
    def __init__(self, method=AlgebraicDistanceIndex, minmax=False, norm=False):
        super().__init__(method, minmax=minmax, norm=norm)


class KatzCalc(Calculate):
    def __init__(self, method=KatzIndex, minmax=False, norm=False):
        super().__init__(method, minmax=minmax, norm=norm)


class NetworkitSparsificationCalculate(Calculate):
    def __init__(self, method, minmax=False, norm=False, triangles=False):
        super().__init__(method, minmax=minmax, norm=norm)
        self.triangles = triangles

    def run(self, graph: Graph):
        if self.triangles:
            calculated_score = self._method(graph, get_triangles(graph))
        else:
            calculated_score = self._method(graph)
        calculated_score.run()
        scores = calculated_score.scores()
        scores = np.array(scores, dtype=np.float32)
        scores = scores[np.isfinite(scores)]
        if self._norm:
            scores = (scores - np.mean(scores)) / (np.std(scores) + np.finfo(np.float32).eps)
        return scores


class LDSCalc(NetworkitSparsificationCalculate):
    def __init__(self, method=LocalDegreeScore):
        super().__init__(method, triangles=False)


class LSSCalc(NetworkitSparsificationCalculate):
    def __init__(self, method=LocalSimilarityScore):
        super().__init__(method, triangles=True)


# class FFSCalc(Calculate):
#     def __init__(self, method=ForestFireScore, minmax=False, norm=False):
#         super().__init__(method, minmax=minmax, norm=norm)
#
#     def run(self, graph: Graph) -> np.ndarray:
#         forest_fire = self._method(graph, 0.5, 1.0)
#         forest_fire.run()
#         scores = forest_fire.scores()
#         scores = np.array(scores, dtype=np.float32)
#         scores = scores[np.isfinite(scores)]
#         if self._minmax:
#             scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + np.finfo(np.float32).eps)
#         if self._norm:
#             scores = (scores - np.mean(scores)) / (np.std(scores) + np.finfo(np.float32).eps)
#         return scores
