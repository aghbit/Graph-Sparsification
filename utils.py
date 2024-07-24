import numpy as np
import matplotlib.pyplot as plt
import torch
from sparsing.sparsing_algorithms import *


def show_cdf(data, percentile_threshold=95):
    sorted_data = np.sort(data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    threshold_value = np.percentile(data, percentile_threshold)

    plt.plot(sorted_data, cdf_values, marker='.', linestyle='none', color='skyblue')
    plt.xlabel('Value')
    plt.xscale('log')

    plt.ylabel('Cumulative Probability')
    plt.title(f'Cumulative Distribution Function (Threshold at {percentile_threshold}%: {threshold_value:.2f})')

    plt.axhline(y=percentile_threshold / 100, color='red', linestyle='dashed', linewidth=1, label=f'{percentile_threshold}% Threshold')
    plt.axvline(x=threshold_value, color='red', linestyle='dashed', linewidth=1)

    plt.legend()
    print(f"Value at {percentile_threshold}% threshold: {threshold_value:.2f}")
    plt.show()

def generate_torch_masks(data, train_ratio=0.8):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_size = int(num_nodes * train_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    return data
