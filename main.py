import warnings

import pandas as pd
import torch

from datasets import datasets
from experiment import run_exp, ExperimentDto
from models import models
from sparsing.sparsing_algorithms import sparsing_list, powers

warnings.filterwarnings('ignore')

device = torch.device('cpu')# "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

from dataclasses import dataclass, field
from typing import Optional
import torch
from result import Result


def get_model(model_type, dataset):
    try:
        model = model_type(dataset)
    except TypeError:
        model = model_type(dataset.num_node_features, dataset.num_node_features, 3, dataset.num_classes)
    return model.to(device)


if __name__ == '__main__':
    run_num = 10
    results = []
    for dataset in datasets:
        for model_data in models:
            for sparsing in sparsing_list:
                sparsing_name = sparsing.__name__ if sparsing is not None else 'NoSparsification'
                try:
                    for power in powers[dataset.name][sparsing_name]:
                        sparsing_alg = sparsing if power is None else sparsing(power)
                        torch.manual_seed(42)
                        torch.cuda.manual_seed_all(42)
                        model = get_model(model_data.model_type, dataset)
                        experiment_dto = ExperimentDto(dataset, model, sparsing_alg)
                        run_results = [run_exp(experiment_dto) for _ in range(run_num)]
                        acc, removed_percentages = zip(*run_results)
                        removed_percentage = removed_percentages[0]

                        result = Result(model_data.model_name, dataset.name, sparsing_name, acc, power,
                                        removed_percentage)
                        print(result)
                        results.append(result.as_dict())

                except KeyError:
                    result = Result(model_data.model_name, dataset.name, sparsing_name, None, None, None)
                    results.append(result.as_dict())
                    print(f'{model_data.model_name} on {dataset} with {sparsing_name} sparsing: No powers found')
    results_df = pd.DataFrame(results)
    results_df.to_csv('additional_files/results.csv', index=False)
