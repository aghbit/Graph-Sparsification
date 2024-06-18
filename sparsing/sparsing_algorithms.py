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


powers = {}
powers['Cora'] = {}
powers['Cora']['NoSparsification'] = [None]
powers['Cora']['PreferentialAttachment'] = [(1e-4) / 2, (1e-3) / 5, (1e-3) / 1.7, (1e-3) / 1.5]
powers['Cora']['AdjustedRand'] = [0.02, 0.06, 0.08, 0.105]
powers['Cora']['Katz'] = [(1e-15), (1e-4), (1e-3) / 3]
powers['Cora']['LDS'] = [1e-15] #removes 9.21% minimum
powers['cora'] = {}
powers['cora']['NoSparsification'] = [None]
powers['CiteSeer'] = {}
powers['CiteSeer']['NoSparsification'] = [None]
powers['CiteSeer']['PreferentialAttachment'] = [1e-15, 1e-3 / 2] #removes 5.47% minimum
powers['CiteSeer']['AdjustedRand'] = [0.02, 0.04, 0.05, 0.06, 0.065]
powers['CiteSeer']['Katz'] = [1e-15, 1e-3 / 4] #removes 6.77% minimum
powers['CiteSeer']['LDS'] = [1e-15, 0.01, 0.15] #removes 7.29% minimum

powers['PubMed'] = {}
powers['PubMed']['NoSparsification'] = [None]
powers['PubMed']['PreferentialAttachment'] = [1e-5, 1e-4, (1e-3)/5, (1e-3)/3]
powers['PubMed']['AdjustedRand'] = [1e-2, 0.015, 0.02, 0.021, 0.0215]
powers['PubMed']['Katz'] = [1e-15, 1e-6, 1e-4] #removes minimum 1.10%
powers['PubMed']['LDS'] = [1e-3, 0.04, 0.06, 0.08, 0.1]

powers['Physics'] = {}
# powers['Physics']['NoSparsification'] = [None]
# powers['Physics']['Jaccard'] = [1e-15, 0.015, 0.02] #removes minimum 4.42%
# powers['Physics']['CommonNeighbor'] = [1e-15] #removes minimum 4.42%
# powers['Physics']['PreferentialAttachment'] = [1e-4, (1e-3)/5, (1e-3)/3]
# powers['Physics']['AdamicAdar'] = [1e-15, 0.005, 0.006] #removes minimum 4.42%
# powers['Physics']['AdjustedRand'] = [0.02, 0.021, 0.0215, 0.022, 0.025]
# powers['Physics']['Katz'] = [1e-4] #0.84%
# powers['Physics']['Katz'] = [1e-3]
# powers['Physics']['LDS'] = [1e-15, 0.04, 0.06, 0.08, 0.1] #removes minimum 2.12%

powers['CS'] = {}
# powers['CS']['NoSparsification'] = [None]
# #powers['CS']['Jaccard'] = [1e-15] #removes minimum 12.46%
# #powers['CS']['CommonNeighbor'] = [1e-15] #removes minimum 12.46%
# powers['CS']['PreferentialAttachment'] = [(1e-3)/5, (1e-3)/3, (1e-3)/2, (1e-3)/1.5, 1e-3]
# #powers['CS']['AdamicAdar'] = [1e-15] #removes minimum 12.46%
# powers['CS']['AdjustedRand'] = [0.011, 0.013, 0.014, 0.015, 0.0155, 0.016]
# powers['CS']['Katz'] = [(1e-3)/6 (1e-3)/4, (1e-3)/2, (1e-3)/1.5, 1e-3]
# powers['CS']['LDS'] = [1e-15, 0.05, 0.08] #removes minimum 4.10%

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