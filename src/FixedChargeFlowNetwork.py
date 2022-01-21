import os

import networkx as nx
from pyvis.network import Network as netVis

from src.Edge import Edge
from src.Node import Node


class FixedChargeFlowNetwork:
    """Class that defines a Fixed Charge Flow Network"""

    def __init__(self):
        """Constructor of a FCFN instance with a NetworkX instance and two data dictionaries"""
        self.name = ""
        self.network = nx.DiGraph()
        self.pipelineCapacities = []
        self.numNodes = 0
        self.numSources = 0
        self.numSinks = 0
        self.nodesDict = {}
        self.numEdges = 0
        self.edgesDict = {}
        self.edgesMap = {}

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
            # Construct source node objects and add to dictionary and network
            if data[0][0] == "s":
                thisNode = Node(data[0], int(data[1]), int(data[2]))
                self.nodesDict[data[0]] = thisNode
                self.network.add_node(data[0])
                self.numSources += 1
            # Construct sink node objects and add to dictionary and network
            if data[0][0] == "t":
                thisNode = Node(data[0], int(data[1]), int(data[2]))
                self.nodesDict[data[0]] = thisNode
                self.network.add_node(data[0])
                self.numSinks += 1
            # Construct transshipment node objects and add to dictionary and network
            if data[0][0] == "n":
                thisNode = Node(data[0], 0, 0)
                self.nodesDict[data[0]] = thisNode
                self.network.add_node(data[0])
            # Construct edge objects and add to dictionary and network
            if data[0][0] == "e":
                # TODO - Account for parallel edges with differing capacities
                thisEdge = Edge(data[0], data[1], data[2], int(data[3]), int(data[4]), int(self.pipelineCapacities[0]))
                edgeKey = (data[1], data[2])
                self.edgesMap[data[0]] = edgeKey
                self.edgesDict[edgeKey] = thisEdge
                self.network.add_edge(data[1], data[2])
                self.nodesDict[data[1]].outgoingEdges.append(data[0])
                self.nodesDict[data[2]].incomingEdges.append(data[0])
        # Assign network size
        self.numNodes = len(self.nodesDict)
        self.numEdges = len(self.edgesDict)

    def drawFCNF(self):
        """Displays the FCNF using PyVis"""
        visual = netVis("500px", "500px", directed=True)
        visual.from_nx(self.network)
        visual.show(str(self.name) + ".html")

    def printAllNodeData(self):
        """Prints all the data for each node in the network"""
        for node in self.nodesDict:
            thisNode = self.nodesDict[node]
            thisNode.printNodeData()

    def printAllEdgeData(self):
        """Prints all the data for each edge in the network"""
        for edge in self.edgesDict:
            thisEdge = self.edgesDict[edge]
            thisEdge.printEdgeData()
