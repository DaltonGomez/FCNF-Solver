import os

import networkx as nx
from docplex.mp.model import Model
from pyvis.network import Network as netVis

from src.Edge import Edge
from src.Node import Node


class FixedChargeFlowNetwork:
    """Class that defines a Fixed Charge Flow Network"""

    def __init__(self):
        """Initializes a FCFN with a NetworkX instance"""
        self.name = ""
        self.network = nx.DiGraph()
        self.pipelineCapacities = []
        self.nodesDict = {}
        self.edgesDict = {}

    def loadFCFN(self, network: str):
        """Loads a FCFN from a text file encoding"""
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
        self.pipelineCapacities = lines[0].split()
        self.pipelineCapacities.pop(0)
        # Build network
        for line in lines:
            data = line.split()
            # Construct sink and source node objects and add to dictionary and network
            if data[0][0] == "s" or data[0][0] == "t":
                thisNode = Node(data[0], int(data[1]), int(data[2]))
                self.nodesDict[data[0]] = thisNode
                self.network.add_node(data[0])
            # Construct transshipment node objects and add to dictionary and network
            if data[0][0] == "n":
                thisNode = Node(data[0], 0, 0)
                self.nodesDict[data[0]] = thisNode
                self.network.add_node(data[0])
            # Construct edge objects and add to dictionary and network
            if data[0][0] == "e":
                # TODO - Account for parallel edge capacities
                thisEdge = Edge(data[0], data[1], data[2], int(data[3]), int(data[4]), int(self.pipelineCapacities[0]))
                edgeKey = (data[1], data[2])
                self.edgesDict[edgeKey] = thisEdge
                self.network.add_edge(data[1], data[2])
        # Test prints
        self.printAllNodeData()
        self.printAllEdgeData()

    def drawFCNF(self):
        """Displays the FCNF using PyVis"""
        visual = netVis("500px", "500px", directed=True)
        # Populates the nodes and edges data structures
        visual.from_nx(self.network)
        visual.show(str(self.name) + ".html")

    def printAllNodeData(self):
        """Prints all the data for each node in the network"""
        for node in self.network.nodes:
            thisNode = self.nodesDict[node]
            thisNode.printNodeData()

    def printAllEdgeData(self):
        """Prints all the data for each edge in the network"""
        for edge in self.network.edges:
            thisEdge = self.edgesDict[edge]
            thisEdge.printEdgeData()

    def solveFCNF(self, targetFlow: int):
        """Solves the FCNF instance via a reduction to a MILP solved in CPLEX"""
        m = Model(name='single variable')
        x = m.binary_var(name="x")
        c1 = m.add_constraint(x >= 2, ctname="const1")
        m.set_objective("min", 3 * x)
        m.print_information()
        m.solve()
        m.print_solution()


# Test Driver
FCFN = FixedChargeFlowNetwork()
FCFN.loadFCFN("small")
FCFN.drawFCNF()
# FCFN.solveFCNF(5)
