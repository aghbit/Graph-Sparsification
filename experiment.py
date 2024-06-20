from dataclasses import dataclass
from typing import Any

import torch
import torch_geometric.datasets as datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class ExperimentDto:
    dataset: datasets
    model: Any
    sparsing_alg: Any


def run_exp(experiment_dto: ExperimentDto):
    dataset, model, sparsing_alg = experiment_dto.dataset, experiment_dto.model, experiment_dto.sparsing_alg
    device = torch.device('cpu')
    data = dataset[0].to(device)

    removed_percentage = None
    if sparsing_alg is not None:
        data, removed_percentage = sparsing_alg.f(data)

    embeddings = model(data.x, data.edge_index)
    X = embeddings
    y = data.y

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)

    return acc, removed_percentage
