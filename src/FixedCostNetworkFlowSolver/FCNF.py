import os

from src.FixedCostNetworkFlowSolver.Edge import Edge
from src.FixedCostNetworkFlowSolver.Node import Node


class FCNF:
    """Class that defines a Fixed Charge Network Flow with parallel edges possible"""

    def __init__(self):
        """Constructor of a FCNF instance"""
        # Input Network Attributes
        self.name = ""
        self.edgeCaps = []
        self.numEdgeCaps = 0
        self.edgeFixedCosts = []
        self.edgeVariableCosts = []
        self.numNodes = 0
        self.numSources = 0
        self.numSinks = 0
        self.nodesDict = {}
        self.numEdges = 0
        self.edgesDict = {}

        # Deterministic seed for consistent visualization
        self.visSeed = 1

        # Solution Data
        self.solved = False
        self.minTargetFlow = 0
        self.totalCost = 0
        self.totalFlow = 0

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
        lines.pop(0)
        self.name = lines[0].split()
        self.name.pop(0)
        self.name = self.name.pop(0)
        lines.pop(0)
        # Assign seed
        self.visSeed = lines[0].split()
        self.visSeed.pop(0)
        self.visSeed = self.visSeed.pop(0)
        lines.pop(0)
        # Assign potential edge capacities
        self.edgeCaps = lines[0].split()
        self.edgeCaps.pop(0)
        lines.pop(0)
        # Assign potential edge fixed costs
        self.edgeFixedCosts = lines[0].split()
        self.edgeFixedCosts.pop(0)
        lines.pop(0)
        # Assign potential edge variable costs
        self.edgeVariableCosts = lines[0].split()
        self.edgeVariableCosts.pop(0)
        lines.pop(0)
        # Build network
        for line in lines:
            data = line.split()
            # Ignore comments
            if data[0][0] == "#":
                continue
            # Construct source node objects and add to dictionary and network
            elif data[0][0] == "s":
                thisNode = Node(data[0], int(data[1]), int(data[2]))
                self.nodesDict[data[0]] = thisNode
                self.numSources += 1
            # Construct sink node objects and add to dictionary and network
            elif data[0][0] == "t":
                thisNode = Node(data[0], int(data[1]), int(data[2]))
                self.nodesDict[data[0]] = thisNode
                self.numSinks += 1
            # Construct transshipment node objects and add to dictionary and network
            elif data[0][0] == "n":
                thisNode = Node(data[0], 0, 0)
                self.nodesDict[data[0]] = thisNode
            # Construct edge objects and add to dictionary and network
            elif data[0][0] == "e":
                thisEdge = Edge(data[0], data[1], data[2])
                self.edgesDict[data[0]] = thisEdge
                self.nodesDict[data[1]].outgoingEdges.append(data[0])
                self.nodesDict[data[2]].incomingEdges.append(data[0])
        # Assign network size
        self.numNodes = len(self.nodesDict)
        self.numEdges = len(self.edgesDict)
        self.numEdgeCaps = len(self.edgeCaps)

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
