from sparsing.calculator import *
from sparsing.abstract_sparsing_algorithm import IndexMain


class Jaccard(IndexMain):
    def __init__(self, power: int = None, calc=JaccardCalc()):
        super(Jaccard, self).__init__(power=power, calc=calc)


class CommonNeighbor(IndexMain):
    def __init__(self, power: int = None, calc=CommonNeighborsCalc()):
        super(CommonNeighbor, self).__init__(power=power, calc=calc)


class PreferentialAttachment(IndexMain):
    def __init__(self, power: int = None, calc=PreferentialAttachmentCalc()):
        super(PreferentialAttachment, self).__init__(power=power, calc=calc)


class AdamicAdar(IndexMain):
    def __init__(self, power: int = None, calc=AdamicAdarCalc()):
        super(AdamicAdar, self).__init__(power=power, calc=calc)


class AdjustedRand(IndexMain):
    def __init__(self, power: int = None, calc=AdjustedRandCalc()):
        super(AdjustedRand, self).__init__(power=power, calc=calc)


class AlgebraicDistance(IndexMain):
    def __init__(self, power: int = None, calc=AlgebraicDistanceCalc()):
        super(AlgebraicDistance, self).__init__(power=power, calc=calc)


class Katz(IndexMain):
    def __init__(self, power: int = None, calc=KatzCalc()):
        super(Katz, self).__init__(power=power, calc=calc)


class LDS(IndexMain):
    def __init__(self, power: int = None, calc=LDSCalc()):
        super().__init__(power=power, calc=calc)


class ForestFire(IndexMain):
    def __init__(self, power: int = None, calc=FFSCalc()):
        super().__init__(power=power, calc=calc)


NORM_TAB = [-3, -2.75, -2.5, -2.25, -2]

sparsing_list = [
    (None, "NoSparsification", [None]),
    # (Random, "Rng", [(i + 1) / 100 for i in range(10)]),
    # (ArithmeticNorm, "ArithmeticNorm", [0.003]),
    # (GeometricNorm, "GeometricNorm", [(i + 1) / 1000 for i in range(20)]),
    # (HarmonicNorm, "HarmonicNorm", [(i + 1) / 1000 for i in range(20)]),
    (Jaccard, "JaccardIndex", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
    (CommonNeighbor, "CommonNeighborIndex", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
    (PreferentialAttachment, "PreferentialAttachment", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
    (AdamicAdar, "AdamicAdar", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
    (AdjustedRand, "AdjustedRand", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
    #(AlgebraicDistance, "AlgebraicDistance", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
    #(Katz, "Katz", [-3, -2.5, -2, -1.5, -1]),
    # (Jaccard, "JaccardIndex", NORM_TAB),
    # (CommonNeighbor, "CommonNeighborIndex", NORM_TAB),
    # (PreferentialAttachment, "PreferentialAttachment", NORM_TAB),
    # (AdamicAdar, "AdamicAdar", NORM_TAB),
    # (AdjustedRand, "AdjustedRand", NORM_TAB),
    #(AlgebraicDistance, "AlgebraicDistance", NORM_TAB),
    (Katz, "Katz", NORM_TAB),
    (LDS, "LocalDegreeScore", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
    (ForestFire, "ForestFire", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
]
