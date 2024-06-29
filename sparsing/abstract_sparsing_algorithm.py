from math import ceil, floor

import torch
from networkit.nxadapter import nx2nk
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx


class BaseSparsing(BaseTransform):
    def __init__(self, power: int = None):
        super(BaseSparsing, self).__init__()
        self.power = power

    def __call__(self, data: Data) -> Data:
        raise NotImplementedError


class IndexMain(BaseSparsing):
    def __init__(self, percent2remove: int = None, calc=None):
        super(IndexMain, self).__init__(power=percent2remove)
        self._calc = calc

    def _main_calc(self, G):
        return self._calc.run(G)

    def __call__(self, data: Data) -> tuple[Data, float]:
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

            edge_index_t = edge_index.T

            edge_weight_pairs = list(zip(edge_index_t, edge_weights))
            edge_weight_pairs.sort(key=lambda x: x[1])
            edge_index_t = torch.stack(list(zip(*edge_weight_pairs))[0], dim=0)

            #edge_index.sort(key = lambda x: edge_weights[edge_index.index(x)])
            n_percent_index = floor(len(edge_weights) * 0.01 * self.power) + 1
            edge_index = edge_index_t.T
            edge_index = edge_index[:, n_percent_index:]

            # sorted_edge_weights = sorted(edge_weights)
            # index_of_one_percent = floor(len(sorted_edge_weights) * 0.01 * self.power)
            # threshold = sorted_edge_weights[index_of_one_percent]
            # edge_index = edge_index[:, edge_weights > threshold]
            after = float(edge_index.shape[1])

            # print(f'Removed {(1 - (after / before)):.2%} of edges')
            edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=1)
            data.edge_index = edge_index
        return data, (1 - (after / before))

    def f(self, data: Data) -> tuple[Data, float]:
        return self.__call__(data)