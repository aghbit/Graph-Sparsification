import torch
import pandas as pd

from classification.models import models
from datasets import datasets
from experiment import run_exp, ExperimentDto
from sparsing.sparsing_algorithms import sparsing_list, powers
import warnings

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')


def get_model(model_type, dataset):
    try:
        model = model_type(dataset)
    except TypeError:
        model = model_type(dataset.num_node_features, dataset.num_node_features, 3, dataset.num_classes)
    return model.to(device)


if __name__ == '__main__':
    run_num = 1
    results = []
    for dataset in datasets:
        for model_data in models:
            for sparsing in sparsing_list:
                sparsing_name = sparsing.__name__ if sparsing is not None else 'NoSparsification'
                try:
                    for power in powers[dataset.name][sparsing_name]:
                        sparsing_alg = sparsing if power is None else sparsing(power)
                        model = get_model(model_data.model_type, dataset)
                        experiment_dto = ExperimentDto(dataset, model, sparsing_alg)
                        acc = [run_exp(experiment_dto) for _ in range(run_num)]

                        result = ({
                            'Model Name': model_data.model_name,
                            'Dataset': dataset.name,
                            'Sparsing Name': sparsing_name,
                            'Power': f'{power:.2g}' if power is not None else 'None',
                            'Accuracy Mean': f'{torch.tensor(acc).mean():.2%}',
                            'Accuracy Std': f'{torch.tensor(acc).std():.2%}'
                        })
                        
                        print(
                            f'{result["Model Name"]} '
                            f'on {result["Dataset"]} '
                            f'with {result["Sparsing Name"]} (power {result['Power']}) '
                            f'sparsing: {result['Accuracy Mean']} '
                            f'± {result['Accuracy Std']}')
                        results.append(result)
                except KeyError:
                    results.append({
                    'Model Name': model_data.model_name,
                    'Dataset': dataset.name,
                    'Sparsing Name': sparsing_name,
                    'Power': None,
                    'Accuracy Mean': None,
                    'Accuracy Std': None,
                    })
                    print(f'{model_data.model_name} on {dataset} with {sparsing_name} sparsing: No powers found')
                    continue
    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)

