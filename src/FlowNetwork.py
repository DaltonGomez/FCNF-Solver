import networkx as nx
import pyvis

# Initialize network
network = nx.DiGraph()

# Test network
network.add_edge(1, 2, capacity=100)
network.add_edge(2, 3, capacity=50)
