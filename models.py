from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import Linear
from torch_geometric.nn import SGConv

@dataclass
class ModelData:
    model_type: Any
    model_name: str

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 200)
        self.conv2 = GCNConv(200, 100)
        self.out = Linear(100, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return self.out(x)

class SAGE(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_node_features, 200)
        self.conv2 = SAGEConv(200, 100)
        self.out = Linear(100, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return self.out(x)

class GAT(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GATConv(dataset.num_node_features, 200)
        self.conv2 = GATConv(200, 100)
        self.out = Linear(100, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return self.out(x)


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
    #ModelData(SGC_CUSTOM, "SGC_CUSTOM"),
    ModelData(GCN, 'GCN'),
    ModelData(SAGE, 'SAGE'),
    ModelData(GAT, 'GAT'),
    # ModelData(GCN, "GCN"),
    # ModelData(GraphSAGE, "GraphSAGE"),
    # ModelData(GIN, "GIN"),
    # ModelData(PNA, "PNA"),
    # ModelData(GAT, "GAT"),
    # ModelData(SVC, 'SVM'),
    # ModelData(RandomForestClassifier, 'RF')
    ]

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc
