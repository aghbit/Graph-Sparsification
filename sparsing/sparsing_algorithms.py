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


powers= {}
powers['Cora'] = {}
powers['Cora']['NoSparsification'] = [None]
powers['Cora']['PreferentialAttachment'] = [(1e-3) / 2, (1e-3) / 1.5, (1e-3) / 1.1, (1e-3)]
powers['Cora']['AdjustedRand'] = [0.08, 0.1, 0.11, 0.12]
powers['Cora']['Katz'] = [(1e-3) / 3, (1e-3) / 1.5]
powers['Cora']['LocalDegreeScore']= [1e-3]
powers['CiteSeer'] = {}
powers['CiteSeer']['NoSparsification'] = [None]
powers['CiteSeer']['PreferentialAttachment'] = [1e-4, (1e-3) / 2]
powers['CiteSeer']['AdjustedRand'] = [0.06, 0.065, 0.068]
powers['CiteSeer']['Katz'] = [1e-15, 1e-3 / 4, 1e-3 / 3]
powers['CiteSeer']['LocalDegreeScore'] = [1e-15, 0.01, 0.15]
powers['CiteSeer']['ForestFire'] = [1e-15]

sparsing_list = [
    None,
    Jaccard,
    CommonNeighbor,
    PreferentialAttachment,
    AdamicAdar,
    AdjustedRand,
    Katz,
    LDS,
    ForestFire,
]
