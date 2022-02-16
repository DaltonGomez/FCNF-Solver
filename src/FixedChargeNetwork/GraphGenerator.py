import os
import random

import networkx as nx

from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class GraphGenerator:
    """Class that generators graphs and saves them to .txt encodes as test instances"""

    def __init__(self, name: str, numNodes: int, edgeProbability: float, numSources: int, numSinks: int,
                 nodeCostBounds: list, nodeCapBounds: list, edgeFixedCostBounds: list, edgeVariableCostBounds: list,
                 edgeCapacityBounds: list, visSeed: int):
        """Constructor of a GraphGenerator instance"""
        self.name = name
        self.visSeed = visSeed
        self.numNodes = numNodes
        self.edgeProbability = edgeProbability
        self.numSources = numSources
        self.numSinks = numSinks
        self.nodeCapBounds = nodeCapBounds
        self.nodeCostBounds = nodeCostBounds
        self.edgeFixedCostBounds = edgeFixedCostBounds
        self.edgeVariableCostBounds = edgeVariableCostBounds
        self.edgeCapacityBounds = edgeCapacityBounds
        self.outputFCFN = FixedChargeFlowNetwork()

        self.network = nx.DiGraph()
        self.nodeMap = {}  # Tracks NX node ordering vs. output node ordering to maintain consistency
        self.generateRandomNetwork()
        self.parseRandomNetwork()

    def generateRandomNetwork(self):
        """Generates a random Fixed Charge Flow Network using NetworkX"""
        self.network = nx.fast_gnp_random_graph(self.numNodes, self.edgeProbability, seed=None, directed=True)
        # Other types of NetworkX random graph generators:
        # self.network = nx.binomial_graph(self.nodes, self.edgeProbability, seed=None, directed=True)
        # self.network = nx.erdos_renyi_graph(self.nodes, self.edgeProbability, seed=None, directed=True)

    def parseRandomNetwork(self):
        """Randomly assigns nodes as sources or sinks up to the input limit"""
        random.seed()
        # Track sources and sinks already assigned
        sourceSinksAssigned = []
        for i in range(self.numNodes):
            sourceSinksAssigned.append(0)
        # Randomly assign all sources
        sourceID = 0
        while self.outputFCFN.numSources < self.numSources:
            randNode = random.randint(0, self.numNodes - 1)
            if sourceSinksAssigned[randNode] == 0:
                sourceSinksAssigned[randNode] = 1
                cost = random.randint(self.nodeCostBounds[0], self.nodeCostBounds[1])
                cap = random.randint(self.nodeCapBounds[0], self.nodeCapBounds[1])
                self.outputFCFN.addNode("s", sourceID, cost, cap)
                self.nodeMap[randNode] = "s" + str(sourceID)
                self.outputFCFN.numSources += 1
                sourceID += 1
            else:
                continue
        # Randomly assign all sinks
        sinkID = 0
        while self.outputFCFN.numSinks < self.numSinks:
            randNode = random.randint(0, self.numNodes - 1)
            if sourceSinksAssigned[randNode] == 0:
                sourceSinksAssigned[randNode] = 1
                cost = random.randint(self.nodeCostBounds[0], self.nodeCostBounds[1])
                cap = random.randint(self.nodeCapBounds[0], self.nodeCapBounds[1])
                self.outputFCFN.addNode("t", sinkID, cost, cap)
                self.nodeMap[randNode] = "t" + str(sinkID)
                self.outputFCFN.numSinks += 1
                sinkID += 1
            else:
                continue
        # Assign all intermediate nodes
        intermediateNodeID = 0
        for n in range(self.numNodes):
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
            # If an edge is from a sink or to a source, do not add the edge
            if fromNode[0] == "t" or toNode[0] == "s":
                continue
            else:
                fixedCost = random.randint(self.edgeFixedCostBounds[0], self.edgeFixedCostBounds[1])
                variableCost = random.randint(self.edgeVariableCostBounds[0], self.edgeVariableCostBounds[1])
                capacity = random.randint(self.edgeCapacityBounds[0], self.edgeCapacityBounds[1])
                self.outputFCFN.addEdge(edgeID, fromNode, toNode, fixedCost, variableCost, capacity)
                fromNodeObj = self.outputFCFN.nodesDict[fromNode]
                fromNodeObj.outgoingEdges.append("e" + str(edgeID))
                toNodeObj = self.outputFCFN.nodesDict[toNode]
                toNodeObj.incomingEdges.append("e" + str(edgeID))
                edgeID += 1
        self.outputFCFN.name = self.name
        self.outputFCFN.visSeed = self.visSeed
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
        # Construct output block by beginning with graph metadata collated in a header
        outputBlock = ["# Network name and visualization seed",
                       "Name= " + self.outputFCFN.name,
                       "VisualSeed= " + str(self.outputFCFN.visSeed),
                       "#", "# numNodes= " + str(self.numNodes),
                       "# edgeProb= " + str(self.edgeProbability),
                       "# numSources= " + str(self.numSources),
                       "# numSinks= " + str(self.numSinks),
                       "# nodeCapBounds= " + str(self.nodeCapBounds),
                       "# nodeCostBounds= " + str(self.nodeCostBounds),
                       "# edgeFixedCostBounds= " + str(self.edgeFixedCostBounds),
                       "# edgeVariableCostBounds= " + str(self.edgeVariableCostBounds),
                       "# edgeCapacityBounds= " + str(self.edgeCapacityBounds),
                       "#", "# Additional comments can be added below with a leading '#'", "#",
                       "# Source nodes of form: <id, variableCost, capacity>"]
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
            outputBlock.append(
                "e" + str(i) + " " + edjObj.fromNode + " " + edjObj.toNode + " " + str(edjObj.fixedCost) + " " + str(
                    edjObj.variableCost) + " " + str(edjObj.capacity))
        # Open file, write lines in output block, and close file
        with open(catPath, "w") as outputFile:
            for line in outputBlock:
                outputFile.write(line)
                outputFile.write('\n')
        outputFile.close()
