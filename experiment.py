from dataclasses import dataclass
from typing import Any

import torch
import torch_geometric.datasets as datasets
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from models import test
from utils import generate_torch_masks


@dataclass
class ExperimentDto:
    dataset: datasets
    model: Any
    sparsing_alg: Any


def run_exp(experiment_dto: ExperimentDto, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dataset, model, sparsing_alg = experiment_dto.dataset, experiment_dto.model, experiment_dto.sparsing_alg
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data = dataset[0].to(device)

    data = generate_torch_masks(data)

    removed_percentage = None
    if sparsing_alg is not None:
        data, removed_percentage = sparsing_alg.f(data)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for _ in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    acc = test(model, data)

    return acc, removed_percentage


def run_exp_SGC(experiment_dto: ExperimentDto, seed: int):
    dataset, model, sparsing_alg = experiment_dto.dataset, experiment_dto.model, experiment_dto.sparsing_alg
    device = torch.device('cpu')
    data = dataset[0].to(device)

    removed_percentage = None
    if sparsing_alg is not None:
        data, removed_percentage = sparsing_alg.f(data)

    embeddings = model(data.x, data.edge_index)
    X = embeddings
    y = data.y

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=seed)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)

    return acc, removed_percentage
