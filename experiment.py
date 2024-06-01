import torch
import torch.nn.functional as F
from tqdm import tqdm


def run_exp(dataset, model, sparsing_alg=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    if sparsing_alg is not None:
        data = sparsing_alg.f(data)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    out_function = lambda x: model(x) \
        if model.__class__.__name__ in ["GCN_CUSTOM", "SGC_CUSTOM"] \
        else model(x.x, x.edge_index)

    for epoch in tqdm(range(200), desc='Model Training'):
        optimizer.zero_grad()
        out = out_function(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc
