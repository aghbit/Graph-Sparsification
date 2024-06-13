from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import torch

from torch_geometric.utils import to_networkx
from networkit.nxadapter import nx2nk

from utils import show_cdf


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
            #print(f'before: {edge_index.shape}')
            before = float(edge_index.shape[1])
            G = to_networkx(data, to_undirected=True)
            G = nx2nk(G)
            G.removeSelfLoops()
            G.indexEdges()  # INDEX EDGES AFTER REMOVING SELF LOOPS, OTHERWISE ERRORS WILL OCCUR
            edge_weights = self._main_calc(G)
            # show_cdf(edge_weights)
            edge_index = edge_index[:, edge_weights >= self.power]
            #edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=1)
            after = float(edge_index.shape[1])
            #print(f'after: {edge_index.shape}')
            print(f'Removed {(1 - (after / before)):.2%} of edges')
            data.edge_index = edge_index
        return data

    def f(self, data: Data) -> Data:
        return self.__call__(data)
