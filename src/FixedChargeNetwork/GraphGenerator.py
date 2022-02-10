import os
import random

import networkx as nx

from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class GraphGenerator:
    """Class that generators graphs and saves them to .txt encodes as test instances"""

    def __init__(self, nodes: int, edgeProbability: float, sources: int, sinks: int, nodeCapLimit: int,
                 nodeCostLimit: int):
        """Constructor of a GraphGenerator instance"""
        self.nodes = nodes
        self.edgeProbability = edgeProbability
        self.sources = sources
        self.sinks = sinks
        self.nodeCapLimit = nodeCapLimit
        self.nodeCostLimit = nodeCostLimit
        self.outputFCFN = FixedChargeFlowNetwork()

        self.network = nx.DiGraph()
        self.nodeMap = {}  # Tracks NX node ordering vs. output node ordering to maintain consistency
        self.generateRandomNetwork()
        self.parseRandomNetwork()

    def generateRandomNetwork(self):
        """Generates a random Fixed Charge Flow Network using NetworkX"""
        self.network = nx.fast_gnp_random_graph(self.nodes, self.edgeProbability, seed=None, directed=True)
        # self.network = nx.binomial_graph(self.nodes, self.edgeProbability, seed=None, directed=True)
        # self.network = nx.erdos_renyi_graph(self.nodes, self.edgeProbability, seed=None, directed=True)

    def parseRandomNetwork(self):
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
                self.outputFCFN.addNode("s", sourceID, cost, cap)
                self.nodeMap[randNode] = "s" + str(sourceID)
                self.outputFCFN.numSources += 1
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
                self.outputFCFN.addNode("t", sinkID, cost, cap)
                self.nodeMap[randNode] = "t" + str(sinkID)
                self.outputFCFN.numSinks += 1
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
                self.outputFCFN.numIntermediateNodes += 1
                intermediateNodeID += 1
            else:
                continue
        # Add edges into FCFN instance and update node topology
        edgeID = 0
        for edge in self.network.edges:
            fromNode = self.nodeMap[edge[0]]
            toNode = self.nodeMap[edge[1]]
            self.outputFCFN.addEdge(edgeID, fromNode, toNode)
            fromNodeObj = self.outputFCFN.nodesDict[fromNode]
            fromNodeObj.outgoingEdges.append("e" + str(edgeID))
            toNodeObj = self.outputFCFN.nodesDict[toNode]
            toNodeObj.incomingEdges.append("e" + str(edgeID))
            edgeID += 1

    def finalizeRandomNetwork(self, name: str, visSeed: int, edgeCaps: list, edgeFixedCost: list,
                              edgeVariableCosts: list):
        """Adds on remaining attributes to complete the FCFN and returns the object"""
        self.outputFCFN.name = name
        self.outputFCFN.visSeed = visSeed
        self.outputFCFN.edgeCaps = edgeCaps
        self.outputFCFN.numEdgeCaps = len(edgeCaps)
        self.outputFCFN.edgeFixedCosts = edgeFixedCost
        self.outputFCFN.edgeVariableCosts = edgeVariableCosts
        self.outputFCFN.numNodes = len(self.outputFCFN.nodesDict)
        self.outputFCFN.numEdges = len(self.outputFCFN.edgesDict)
        return self.outputFCFN

    def visualizeRandomNetwork(self):
        """Draws the randomly generator network with PyVis"""
        self.outputFCFN.visualizeNetwork()

    def saveFCFN(self):
        """Saves an unsolved version of a NetworkX-generated FCFN as a .txt file within the project directory"""
        # Path management
        currDir = os.getcwd()
        networkFile = self.outputFCFN.name + ".txt"
        catPath = os.path.join(currDir, "networks", networkFile)
        print("Saving " + networkFile + " to: " + catPath)
        # Construct output block
        outputBlock = ["# Network name, visualization seed, and parallel edge data", "Name= " + self.outputFCFN.name,
                       "VisualSeed= " + str(self.outputFCFN.visSeed)]
        edgeCapsLine = "EdgeCaps= "
        for edgeCap in self.outputFCFN.edgeCaps:
            edgeCapsLine = edgeCapsLine + str(edgeCap) + " "
        outputBlock.append(edgeCapsLine)
        edgeFCLine = "EdgeFixedCosts= "
        for edgeFC in self.outputFCFN.edgeFixedCosts:
            edgeFCLine = edgeFCLine + str(edgeFC) + " "
        outputBlock.append(edgeFCLine)
        edgeVCLine = "EdgeVariableCosts= "
        for edgeVC in self.outputFCFN.edgeVariableCosts:
            edgeVCLine = edgeVCLine + str(edgeVC) + " "
        outputBlock.append(edgeVCLine)
        outputBlock.append("#")
        outputBlock.append("# Additional comments can be added below with a leading '#'")
        outputBlock.append("#")
        outputBlock.append("# Source nodes of form: <id, variableCost, capacity>")
        for i in range(self.outputFCFN.numSources):
            srcObj = self.outputFCFN.nodesDict["s" + str(i)]
            outputBlock.append("s" + str(i) + " " + str(srcObj.variableCost) + " " + str(srcObj.capacity))
        outputBlock.append("#")
        outputBlock.append("# Sink nodes of form: <id, variableCost, capacity>")
        for i in range(self.outputFCFN.numSinks):
            sinkObj = self.outputFCFN.nodesDict["t" + str(i)]
            outputBlock.append("t" + str(i) + " " + str(sinkObj.variableCost) + " " + str(sinkObj.capacity))
        outputBlock.append("#")
        outputBlock.append("# Intermediate nodes of form: <id>")
        for i in range(self.outputFCFN.numIntermediateNodes):
            outputBlock.append("n" + str(i))
        outputBlock.append("#")
        outputBlock.append("# Edges of form: <id, fromNode, toNode>")
        for i in range(self.outputFCFN.numEdges):
            edjObj = self.outputFCFN.edgesDict["e" + str(i)]
            outputBlock.append("e" + str(i) + " " + edjObj.fromNode + " " + edjObj.toNode)
        # Open file, write lines in output block, and close file
        with open(catPath, "w") as outputFile:
            for line in outputBlock:
                outputFile.write(line)
                outputFile.write('\n')
        outputFile.close()
