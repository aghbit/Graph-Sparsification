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
    def __init__(self, power: int=None, calc=None, calc_name=None):
        super(IndexMain, self).__init__(power=power)
        self._calc = calc
        self._calc_name = calc_name

    def _main_calc(self, G):
        return self._calc.run(G)

    def __call__(self, data: Data, data_name: str, data_type: str) -> Data:
        if self.power is not None:
            edge_index = data.edge_index
            mask = edge_index[0] < edge_index[1]
            edge_index = edge_index[:, mask]
            G = to_networkx(data, to_undirected=True)
            G = nx2nk(G)
            edge_weights = self._main_calc(G)
            #show_cdf(edge_weights, data_name, data_type, self._calc_name)
            edge_index = edge_index[:, edge_weights >= self.power]
            edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=1)
            data.edge_index = edge_index
        return data

    def f(self, data: Data) -> Data:
        return self.__call__(data)
