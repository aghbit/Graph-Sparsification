from dataclasses import dataclass
from typing import Any


from sparsing.calculator import *
from sparsing.abstract_sparsing_algorithm import IndexMain


class Jaccard(IndexMain):
    def __init__(self, power: int = None, calc=JaccardCalc(minmax=True)):
        super(Jaccard, self).__init__(percent2remove=power, calc=calc)


class CommonNeighbor(IndexMain):
    def __init__(self, power: int = None, calc=CommonNeighborsCalc(minmax=True)):
        super(CommonNeighbor, self).__init__(percent2remove=power, calc=calc)


class PreferentialAttachment(IndexMain):
    def __init__(self, power: int = None, calc=PreferentialAttachmentCalc(minmax=True)):
        super(PreferentialAttachment, self).__init__(percent2remove=power, calc=calc)


class AdamicAdar(IndexMain):
    def __init__(self, power: int = None, calc=AdamicAdarCalc(minmax=True)):
        super(AdamicAdar, self).__init__(percent2remove=power, calc=calc)


class AdjustedRand(IndexMain):
    def __init__(self, power: int = None, calc=AdjustedRandCalc(minmax=True)):
        super(AdjustedRand, self).__init__(percent2remove=power, calc=calc)


class AlgebraicDistance(IndexMain):
    def __init__(self, power: int = None, calc=AlgebraicDistanceCalc(minmax=True)):
        super(AlgebraicDistance, self).__init__(percent2remove=power, calc=calc)


class Katz(IndexMain):
    def __init__(self, power: int = None, calc=KatzCalc(minmax=True)):
        super(Katz, self).__init__(percent2remove=power, calc=calc)


class LDS(IndexMain):
    def __init__(self, power: int = None, calc=LDSCalc()):
        super().__init__(percent2remove=power, calc=calc)


class LSS(IndexMain):
    def __init__(self, power: int = None, calc=LSSCalc()):
        super().__init__(percent2remove=power, calc=calc)

class SCAN(IndexMain):
    def __init__(self, power: int = None, calc=SCANCalc()):
        super().__init__(percent2remove=power, calc=calc)

# class ForestFire(IndexMain):
#     def __init__(self, power: int = None, calc=FFSCalc(minmax=True)):
#         super().__init__(percent2remove=power, calc=calc)

sparsing_list = [
    None,
    #Jaccard,
    # CommonNeighbor,
    #PreferentialAttachment,
    #AdamicAdar,
    #AdjustedRand,
    #Katz,
    LDS,
    LSS,
    SCAN,
]
