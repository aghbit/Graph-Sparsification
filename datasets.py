from torch_geometric.datasets import *

datasets = [
    #Planetoid(root='/tmp/Cora', name='Cora'),
    #Planetoid(root='/tmp/CiteSeer', name='CiteSeer'),
    #Planetoid(root='/tmp/PubMed', name='PubMed'),
    # Coauthor(root='/tmp/Physics', name='Physics'),
    # Coauthor(root='/tmp/ComputerScience', name='CS'),
    # CitationFull(root='/tmp/CoraFull', name='Cora'),
    WebKB(root='/tmp/Winsconsin', name='Wisconsin'),
    WebKB(root='/tmp/Texas', name='Texas'),
    WebKB(root='/tmp/Cornell', name='Cornell'),
    #Reddit2(root='/tmp/Reddit'),
    #WikiCS(root='/tmp/WikiCS')
    # Twitch(root='/tmp/TwitchDE', name='DE'),
    # Twitch(root='/tmp/TwitchPT', name='PT'),
    # Twitch(root='/tmp/TwitchES', name='ES'),
    #Amazon(root='/tmp/AmazonPhoto', name='Photo'),
    #Amazon(root='/tmp/AmazonComputers', name='Computers')
    #Amazon(root='/tmp/AmazonPhoto', name='Photo'),
    #Amazon(root='/tmp/AmazonComputers', name='Computers'),
    HeterophilousGraphDataset(root='/tmp/roman_empire', name='Roman-empire'),
    HeterophilousGraphDataset(root='/tmp/amazon_ratings', name='Amazon-ratings'),
    HeterophilousGraphDataset(root='/tmp/minesweeper', name='Minesweeper'),
]
