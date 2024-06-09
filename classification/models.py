import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Linear
from torch_geometric.nn import SGConv
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, PNA, GAT
from sklearnex.ensemble import RandomForestClassifier
from sklearnex.svm import SVC

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


class SGC_CUSTOM(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = SGConv(dataset.num_node_features, dataset.num_node_features, K=2)
        self.conv2 = SGConv(dataset.num_node_features, dataset.num_node_features, K=2)
        self.lin1 = Linear(dataset.dataset.num_node_features, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin1(x)

        return F.log_softmax(x, dim=1)


models = [
    # (GCN_CUSTOM, "GCN_CUSTOM"),
    # (SGC_CUSTOM, "SGC_CUSTOM"),
    # (GCN, "GCN"),
    # (GraphSAGE, "GraphSAGE"),
    # (GIN, "GIN"),
    # (PNA, "PNA"),
    # (GAT, "GAT"),
    (SVC, 'SVM'),
    (RandomForestClassifier, 'RF')
    ]
