from datasets import datasets
from torch_geometric.utils import to_networkx
import networkx as nx
import pandas as pd


def calculate_node_homophily_pyg(data) -> float:
    total_nodes = data.num_nodes
    if total_nodes == 0:
        return 0
    
    homophily_sum = 0

    graph = to_networkx(data)

    for node in range(total_nodes):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 0:
            continue
        
        same_class_count = 0
        
        node_class = data.y[node].item()
        
        for neighbor in neighbors:
            if data.y[neighbor].item() == node_class:
                same_class_count += 1
        
        if len(neighbors) > 0:
            homophily_ratio = same_class_count / len(neighbors)
        else:
            homophily_ratio = 0
        
        homophily_sum += homophily_ratio
    
    node_homophily_index = homophily_sum / total_nodes
    
    return node_homophily_index

results = []

for dataset in datasets:
    homophily_index = calculate_node_homophily_pyg(dataset[0])
    print(f"Node Homophily Index for {dataset.name}: {homophily_index}")
    results.append({'Dataset Name': dataset.name, 'Node Homophily Index': homophily_index})

df = pd.DataFrame(results)

df = df.sort_values(by='Node Homophily Index')

csv_file = 'homophily_indices.csv'

df.to_csv(csv_file, index=False)


print(f"Results saved to {csv_file}")

