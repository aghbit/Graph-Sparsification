from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Linear
from torch_geometric.nn import SGConv

@dataclass
class ModelData:
    model_type: Any
    model_name: str

class GCN_CUSTOM(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dataset.num_node_features)
        self.conv2 = GCNConv(dataset.num_node_features, dataset.num_node_features)
        self.lin1 = Linear(dataset.num_node_features, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin1(x)

        return F.log_softmax(x, dim=1)


# SGC Feature Extractor
class SGC_CUSTOM(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = SGConv(dataset.num_node_features, dataset.num_node_features, K=2)
        self.conv1.lin = torch.nn.Identity()

    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)


models = [
    # ModelData(GCN_CUSTOM, "GCN_CUSTOM"),
    ModelData(SGC_CUSTOM, "SGC_CUSTOM"),
    # ModelData(GCN, "GCN"),
    # ModelData(GraphSAGE, "GraphSAGE"),
    # ModelData(GIN, "GIN"),
    # ModelData(PNA, "PNA"),
    # ModelData(GAT, "GAT"),
    # ModelData(SVC, 'SVM'),
    # ModelData(RandomForestClassifier, 'RF')
    ]
