import torch

from datasets import datasets
from experiment import run_exp, run_exp_sklearn
from classification.models import models
from sparsing.sparsing_algorithms import sparsing_list

import warnings
warnings.filterwarnings('ignore')


def get_model(model_type, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f'Using device: {device}')
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
    run_num = 1

    with open('results_wikiCS.txt', 'w') as f:
        f.write('Experiment results:\n\n')

    for dataset in datasets:
        # with open('results_wikiCS.txt', 'a') as f:
        #     f.write(f'\n{dataset}:\n\n')
        for model_type, model_name in models:
            for algorithm_type, algorithm_name, all_powers in sparsing_list:
                for dataset_name, dataset_powers_dict in all_powers.items():
                    if dataset_name != dataset.name:
                        continue

                    # if model_name in ['RF', 'SVM']:
                    for algorithm_name_power, dataset_algorithm_powers in dataset_powers_dict.items():
                        if algorithm_name_power != algorithm_name:
                            continue
                        for power in dataset_algorithm_powers:
                            acc, precision, recall = zip(*[run_exp_sklearn(
                                dataset,
                                model=model_name,
                                sparsing_alg=algorithm_type if power is None else algorithm_type(power)
                            ) for _ in range(run_num)])
                            # acc = [run_exp(
                            #     dataset,
                            #     model=get_model(model_type, dataset),
                            #     sparsing_alg=algorithm_type if power is None else algorithm_type(power)
                            # ) for _ in range(run_num)]

                            result = (
                                f'{model_name} '
                                f'on {dataset} '
                                f'with {algorithm_name} (power {power}) sparsing: '
                                f'accuracy {torch.tensor(acc).mean():.2%} | '
                                f'± {torch.tensor(acc).std():.2%} | ')
                                # f'precision {torch.tensor(precision).mean():.2%} | '
                                # f'recall {torch.tensor(recall).mean():.2%} | ')

                        print(result)
                    # with open('results_wikiCS.txt', 'a') as f:
                    #     f.write(f'{result}\n')
