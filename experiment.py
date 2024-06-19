from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F
import torch_geometric.datasets as datasets
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@dataclass
class ExperimentDto:
    dataset: datasets
    model: Any
    sparsing_alg: Any


def run_exp(experiment_dto: ExperimentDto):
    dataset, model, sparsing_alg = experiment_dto.dataset, experiment_dto.model, experiment_dto.sparsing_alg
    device = torch.device('cpu') # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)

    removed_percentage = None
    if sparsing_alg is not None:
        data, removed_percentage = sparsing_alg.f(data)
    #optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01, weight_decay=5e-4)
    # model.train()
    #
    # for epoch in range(100):
    #     optimizer.zero_grad()
    #     out = model(data.x, data.edge_index)
    #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #     loss.backward()
    #     optimizer.step()
    # model.eval()

    embeddings = model(data.x, data.edge_index)
    # X_train = embeddings[data.train_mask]
    # X_test = embeddings[data.test_mask]
    # y_train = data.y[data.train_mask]
    # y_test = data.y[data.test_mask]
    X = embeddings
    y = data.y

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)

    # pred = model(data.x, data.edge_index).argmax(dim=1)
    # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    # acc = int(correct) / int(data.test_mask.sum())
    return acc, removed_percentage # acc
