import torch

from datasets import datasets
from experiment import run_exp
from models import models
from sparsing.sparsing_algorithms import sparsing_list


def get_model(model_type, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')
    try:
        model = model_type(dataset)
    except TypeError:
        model = model_type(dataset.num_node_features, dataset.num_node_features, 3, dataset.num_classes)
    return model.to(device)


def call_exp(sparsing_alg, string, dataset):
    acc = [run_exp(
        dataset,
        model=get_model(model_type, dataset),
        sparsing_alg=sparsing_alg
    ) for _ in range(run_num)]
    print(
        f'{model_name} on {dataset} with {string} sparsing: {torch.tensor(acc).mean():.2%} ± {torch.tensor(acc).std():.2%}')


if __name__ == '__main__':
    run_num = 10

    for dataset in datasets[1:]:
        for model_type, model_name in models:
            for algorithm_type, algorithm_name, powers in sparsing_list:
                for power in powers:
                    acc = [run_exp(
                        dataset,
                        model=get_model(model_type, dataset),
                        sparsing_alg=algorithm_type if power is None else algorithm_type(power)
                    ) for _ in range(run_num)]

                    print(
                        f'{model_name} '
                        f'on {dataset} '
                        f'with {algorithm_name} '
                        f'sparsing: {torch.tensor(acc).mean():.2%} '
                        f'± {torch.tensor(acc).std():.2%}')
