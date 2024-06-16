from torch_geometric.datasets import *

datasets = [
    Planetoid(root='/tmp/Cora', name='Cora'),
    # Planetoid(root='/tmp/CiteSeer', name='CiteSeer'),
    # Planetoid(root='/tmp/PubMed', name='PubMed'),
    # Coauthor(root='/tmp/Physics', name='Physics'),
    # Coauthor(root='/tmp/ComputerScience', name='CS'),
    #Reddit2(root='/tmp/Reddit'),
    #WikiCS(root='/tmp/WikiCS')
]
