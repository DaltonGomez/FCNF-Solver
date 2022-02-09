import random

import networkx as nx

from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class GraphGenerator:
    """Class that generators graphs and saves them to .txt encodes as test instances"""

    def __init__(self, nodes: int, edgeProbability: float, sources: int, sinks: int, nodeCapLimit: int,
                 nodeCostLimit: int, name: str):
        """Constructor of a GraphGenerator instance"""
        self.nodes = nodes
        self.edgeProbability = edgeProbability
        self.sources = sources
        self.sinks = sinks
        self.nodeCapLimit = nodeCapLimit
        self.nodeCostLimit = nodeCostLimit
        self.outputFCFN = FixedChargeFlowNetwork()
        self.outputFCFN.name = name

        self.network = nx.DiGraph()
        self.nodeMap = {}  # Tracks NX node ordering vs. output node ordering to maintain consistency
        self.generateRandomNetwork()
        self.assignSourceSinksAndEdges()

    def generateRandomNetwork(self):
        """Generates a random Fixed Charge Flow Network using NetworkX"""
        self.network = nx.fast_gnp_random_graph(self.nodes, self.edgeProbability, seed=None, directed=True)

    def assignSourceSinksAndEdges(self):
        """Randomly assigns nodes as sources or sinks up to the input limit"""
        random.seed()
        # Track sources and sinks already assigned
        sourceSinksAssigned = []
        for i in range(self.nodes):
            sourceSinksAssigned.append(0)
        # Randomly assign all sources
        sourceID = 0
        while self.outputFCFN.numSources < self.sources:
            randNode = random.randint(0, self.nodes - 1)
            if sourceSinksAssigned[randNode] == 0:
                sourceSinksAssigned[randNode] = 1
                cap = random.randint(0, self.nodeCapLimit)
                cost = random.randint(0, self.nodeCostLimit)
                self.outputFCFN.addNode("s", sourceID, cap, cost)
                self.nodeMap[randNode] = "s" + str(sourceID)
                sourceID += 1
            else:
                continue
        # Randomly assign all sinks
        sinkID = 0
        while self.outputFCFN.numSinks < self.sinks:
            randNode = random.randint(0, self.nodes - 1)
            if sourceSinksAssigned[randNode] == 0:
                sourceSinksAssigned[randNode] = 1
                cap = random.randint(0, self.nodeCapLimit)
                cost = random.randint(0, self.nodeCostLimit)
                self.outputFCFN.addNode("t", sinkID, cap, cost)
                self.nodeMap[randNode] = "t" + str(sinkID)
                sinkID += 1
            else:
                continue
        # Assign all intermediate nodes
        intermediateNodeID = 0
        for n in range(self.nodes):
            if sourceSinksAssigned[n] == 0:
                sourceSinksAssigned[n] = 1
                self.outputFCFN.addNode("n", intermediateNodeID, 0, 0)
                self.nodeMap[n] = "n" + str(intermediateNodeID)
                intermediateNodeID += 1
            else:
                continue
        # Add edges into FCFN instance and update node topology
        edgeID = 0
        for edge in self.network.edges:
            fromNode = self.nodeMap[edge[0]]
            toNode = self.nodeMap[edge[1]]
            self.outputFCFN.addEdge(edgeID, fromNode, toNode)
            edgeID += 1
            fromNodeObj = self.outputFCFN.nodesDict[fromNode]
            fromNodeObj.outgoingEdges.append("e" + str(edgeID))
            toNodeObj = self.outputFCFN.nodesDict[toNode]
            toNodeObj.outgoingEdges.append("e" + str(edgeID))

    def drawRandomGraph(self):
        """Draws the randomly generator network with PyVis"""
        self.outputFCFN.visualizeNetwork()

    def saveFCFNtoDisc(self):
        """Saves an unsolved version of a NetworkX-generated FCFN as a .txt file within the project directory"""
        # TODO - Implement
        pass
