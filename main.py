import torch

from classification.models import models
from datasets import datasets
from experiment import run_exp, ExperimentDto
from sparsing.sparsing_algorithms import sparsing_list

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')
import warnings
warnings.filterwarnings('ignore')


def get_model(model_type, dataset):
    try:
        model = model_type(dataset)
    except TypeError:
        model = model_type(dataset.num_node_features, dataset.num_node_features, 3, dataset.num_classes)
    return model.to(device)


if __name__ == '__main__':
    run_num = 10
    for dataset in datasets:
        for model_data in models:
            for sparsing_data in sparsing_list:
                for power in sparsing_data.powers:

                    sparsing_alg = sparsing_data.algorithm_type if power is None else sparsing_data.algorithm_type(power)
                    model = get_model(model_data.model_type, dataset)
                    experiment_dto = ExperimentDto(dataset, model, sparsing_alg)
                    acc = [run_exp(experiment_dto) for _ in range(run_num)]

                    print(
                        f'{model_data.model_name} '
                        f'on {dataset} '
                        f'with {sparsing_data.algorithm_name} (power {power}) '
                        f'sparsing: {torch.tensor(acc).mean():.2%} '
                        f'± {torch.tensor(acc).std():.2%}')
