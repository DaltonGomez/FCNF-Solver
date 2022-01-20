import os
import networkx as nx
from pyvis.network import Network as netVis

class FixedChargeFlowNetwork:
    """Class that defines a Fixed Charge Flow Network."""

    def __init__(self):
        """Initializes a FCFN with a NetworkX instance."""
        self.name = ""
        self.network = nx.DiGraph()
        self.capacities = []

    def loadFCFN(self, network: str):
        """Loads a FCFN from a text file encoding."""
        # Path management
        currDir = os.getcwd()
        networkFile = network + ".txt"
        catPath = os.path.join(currDir, "networks", networkFile)
        print("Loading " + networkFile + " from: " + catPath)
        # Open file, parse lines in data stream, and close file
        inputFile = open(catPath, 'r')
        lines = inputFile.readlines()
        inputFile.close()
        # Assign name
        self.name = lines[0].rstrip()
        lines.pop(0)
        # Assign capacities
        self.capacities = lines[0].split()
        self.capacities.pop(0)
        # Build network
        for line in lines:
            data = line.split()
        # Construct sources
            if data[0][0] == "s":
                self.network.add_node(data[0], fixedCost=data[1], variableCost=data[2])
        # Construct sinks
            if data[0][0] == "t":
                self.network.add_node(data[0], fixedCost=data[1], variableCost=data[2])
        # Construct intermediate nodes
            if data[0][0] == "n":
                self.network.add_node(data[0])
        # Construct edges
            if data[0][0] == "e":
                self.network.add_edge(data[1], data[2], fixedCost=data[3], variableCost=data[4])
        # Test prints
        print(self.network.nodes)
        print(self.network.edges)
        print(self.capacities)

    def drawFCNF(self):
        """Displays the FCNF using PyVis"""
        visual = netVis("500px", "500px", directed=True)
        # populates the nodes and edges data structures
        visual.from_nx(self.network)
        visual.show(str(self.name) + ".html")



# Test Driver
FCFN = FixedChargeFlowNetwork()
FCFN.loadFCFN("small")
FCFN.drawFCNF()
