import numpy as np
import torch
import torch.nn.functional as F
from sklearnex.ensemble import RandomForestClassifier
from sklearnex.svm import SVC
from sklearnex.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from classification.feature_extraction import get_topological_features
from torch_geometric.utils import to_networkx
import networkit as nk


def run_exp(dataset, model, sparsing_alg=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    if sparsing_alg is not None:
        data = sparsing_alg.f(data)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc


# TODO handle both torch and sklearn models in a single run_exp function?
def run_exp_sklearn(dataset, model, sparsing_alg=None):
    data = dataset[0]

    # apply sparsification (optional)
    if sparsing_alg is not None:
        data = sparsing_alg.f(data)

    # extract topological features from the graphs
    G_nx = to_networkx(data, to_undirected=True)
    G_nk = nk.nxadapter.nx2nk(G_nx)

    # X = data.x.numpy()
    # X = np.hstack((X, get_topological_features(G_nk)))
    X = get_topological_features(G_nk)
    y = data.y.numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # get model
    if model == 'SVM':
        model = SVC(
            kernel='rbf',
            #random_state=42
        )
    elif model == 'RF':
        model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            #random_state=42
        )
    else:
        raise Exception(f'Model {model} not supported')

    # train and evaluate the model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    #print(f'precision: {np.round(precision, 2)}')
    recall = recall_score(y_test, y_pred, average=None)
    #print(f'recall: {np.round(recall, 2)}')
    return acc, precision, recall
