from dataclasses import dataclass
from typing import Any

from sparsing.calculator import *
from sparsing.abstract_sparsing_algorithm import IndexMain


class Jaccard(IndexMain):
    def __init__(self, power: int = None, calc=JaccardCalc(minmax=True)):
        super(Jaccard, self).__init__(power=power, calc=calc)


class CommonNeighbor(IndexMain):
    def __init__(self, power: int = None, calc=CommonNeighborsCalc(minmax=True)):
        super(CommonNeighbor, self).__init__(power=power, calc=calc)


class PreferentialAttachment(IndexMain):
    def __init__(self, power: int = None, calc=PreferentialAttachmentCalc(minmax=True)):
        super(PreferentialAttachment, self).__init__(power=power, calc=calc)


class AdamicAdar(IndexMain):
    def __init__(self, power: int = None, calc=AdamicAdarCalc(minmax=True)):
        super(AdamicAdar, self).__init__(power=power, calc=calc)


class AdjustedRand(IndexMain):
    def __init__(self, power: int = None, calc=AdjustedRandCalc(minmax=True)):
        super(AdjustedRand, self).__init__(power=power, calc=calc)


class AlgebraicDistance(IndexMain):
    def __init__(self, power: int = None, calc=AlgebraicDistanceCalc(minmax=True)):
        super(AlgebraicDistance, self).__init__(power=power, calc=calc)


class Katz(IndexMain):
    def __init__(self, power: int = None, calc=KatzCalc(minmax=True)):
        super(Katz, self).__init__(power=power, calc=calc)


class LDS(IndexMain):
    def __init__(self, power: int = None, calc=LDSCalc(minmax=True)):
        super().__init__(power=power, calc=calc)


class ForestFire(IndexMain):
    def __init__(self, power: int = None, calc=FFSCalc(minmax=True)):
        super().__init__(power=power, calc=calc)


NORM_TAB = [-3, -2.75, -2.5, -2.25, -2]

@dataclass
class SparsingData:
    algorithm_type: Any
    algorithm_name: str
    powers: list


powers = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
sparsing_list = [
    SparsingData(None, "NoSparsification", [None]),
    SparsingData(Jaccard, "JaccardIndex", powers),
    SparsingData(CommonNeighbor, "CommonNeighborIndex", powers),
    SparsingData(PreferentialAttachment, "PreferentialAttachment", powers),
    SparsingData(AdamicAdar, "AdamicAdar", powers),
    SparsingData(AdjustedRand, "AdjustedRand", powers),
    SparsingData(Katz, "Katz", powers),
    SparsingData(LDS, "LocalDegreeScore", powers),
    SparsingData(ForestFire, "ForestFire", powers),
]
