import networkit as nk
import numpy as np
from networkit.centrality import (
    DegreeCentrality,
    LocalClusteringCoefficient,
    Betweenness,
    EigenvectorCentrality,
)


def calculate_centrality_feature(G: nk.Graph, feature: nk.centrality.Centrality) -> np.ndarray:
    return feature(G).run().scores()

def get_topological_features(G: nk.Graph) -> np.ndarray:
    """
    :param G: a graph to extract node features from
    :return: N x F matrix of nodes topological features (for N nodes in a graph and F features)

    Features extracted:
    - Degree Centrality
    - Local Clustering Coefficient
    - Betweenness Centrality
    - Eigenvector Centrality
    """
    G.removeSelfLoops() # temporarily needed for WikiCS dataset
    degree_centrality = calculate_centrality_feature(G, DegreeCentrality)
    local_clustering_coefficient = calculate_centrality_feature(G, LocalClusteringCoefficient)
    betweenness_centrality = calculate_centrality_feature(G, Betweenness)
    eigenvector_centrality = calculate_centrality_feature(G, EigenvectorCentrality)

    return np.array([
        degree_centrality,
        local_clustering_coefficient,
        betweenness_centrality,
        eigenvector_centrality
        ]
    ).T
