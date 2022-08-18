import networkx as nx

g = nx.Graph(name="test")

g.add_node("S", supply="10")
g.add_node("T", demand="10")

g.add_edge("S", "T", cost=5)

print(g)
print(g.name)
print(g.nodes)
print(g.edges)
