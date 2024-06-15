import torch
from networkit.nxadapter import nx2nk
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx


class BaseSparsing(BaseTransform):
    def __init__(self, power: float = None):
        super(BaseSparsing, self).__init__()
        self.power = power

    def __call__(self, data: Data) -> Data:
        raise NotImplementedError


class IndexMain(BaseSparsing):
    def __init__(self, power: int = None, calc=None):
        super(IndexMain, self).__init__(power=power)
        self._calc = calc

    def _main_calc(self, G):
        return self._calc.run(G)

    def __call__(self, data: Data) -> Data:
        if self.power is not None:
            edge_index = data.edge_index
            mask = edge_index[0] < edge_index[1]
            edge_index = edge_index[:, mask]
            before = float(edge_index.shape[1])
            G = to_networkx(data, to_undirected=True)
            G = nx2nk(G)
            G.removeSelfLoops()
            G.indexEdges()
            edge_weights = self._main_calc(G)
            edge_index = edge_index[:, edge_weights >= self.power]
            after = float(edge_index.shape[1])
            print(f'Removed {(1 - (after / before)):.2%} of edges')
            data.edge_index = edge_index
        return data

    def f(self, data: Data) -> Data:
        return self.__call__(data)
