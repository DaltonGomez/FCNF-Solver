import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.spatial import Delaunay

from src.Network.Arc import Arc
from src.Network.Node import Node


class FlowNetwork:
    """Class that defines a Network object with multi-source/multi-sink and parallel edges, called arcs"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self):
        """Constructor of a Flow Network instance"""
        self.name = ""
        # Node Attributes
        self.numTotalNodes = 0
        self.nodesDict = {}
        self.points = None
        self.numSources = 0
        self.sourcesArray = None
        self.numSinks = 0
        self.sinksArray = None
        self.numInterNodes = 0
        self.interNodesArray = None
        # Edge/Arc Attributes
        self.numEdges = 0
        self.edgesArray = None
        self.edgesDict = {}
        self.distancesArray = None
        self.numArcCaps = 0
        self.possibleArcCapsArray = None
        self.numArcs = 0
        self.arcsDict = {}
        self.arcsMatrix = None
        # Capacitated/Charged Source/Sink Generalization
        self.isSourceSinkCapacitated = False
        self.sourceCapsArray = None
        self.sinkCapsArray = None
        # Variable Cost Source/Sink Generalization
        self.isSourceSinkCharged = False
        self.sourceVariableCostsArray = None
        self.sinkVariableCostsArray = None

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def drawNetworkTriangulation(self) -> None:
        """Draws the Delaunay triangulation of the network with MatPlotLib for quick judgement of topology"""
        triangulation = Delaunay(self.points)
        plt.triplot(self.points[:, 0], self.points[:, 1], triangulation.simplices)
        plt.plot(self.points[:, 0], self.points[:, 1], 'ko')
        sourceList = []
        for source in self.sourcesArray:
            thisPoint = self.getNodeCoordinates(source)
            sourceList.append(thisPoint)
        sourcePoints = np.array(sourceList)
        plt.plot(sourcePoints[:, 0], sourcePoints[:, 1], 'yD')
        sinkList = []
        for sink in self.sinksArray:
            thisPoint = self.getNodeCoordinates(sink)
            sinkList.append(thisPoint)
        sinkPoints = np.array(sinkList)
        plt.plot(sinkPoints[:, 0], sinkPoints[:, 1], 'rs')
        # Save figure and display
        currDir = os.getcwd()
        networkFile = self.name + ".png"
        catPath = os.path.join(currDir, "../network_instances", networkFile)
        plt.savefig(catPath)
        plt.show()

    # =================================================
    # ============== DATA IN/OUT METHODS ==============
    # =================================================
    def saveNetwork(self) -> None:
        """Saves the network instance to disc via a pickle dump"""
        # Path management
        currDir = os.getcwd()
        networkFile = self.name + ".p"
        catPath = os.path.join(currDir, "../network_instances", networkFile)
        print("Saving " + networkFile + " to: " + catPath)
        # Pickle dump
        pickle.dump(self, open(catPath, "wb"))

    @staticmethod
    def loadNetwork(networkFile: str):
        """Loads a network instance via a pickle load"""
        # Path management
        currDir = os.getcwd()
        catPath = os.path.join(currDir, "../network_instances", networkFile)
        print("Loading " + networkFile + " from: " + catPath)
        # Pickle load
        flowNetwork = pickle.load(open(catPath, "rb"))
        return flowNetwork

    # =====================================================
    # ============== NETWORK BUILDER METHODS ==============
    # =====================================================
    def addNodeToDict(self, nodeID: int, xPos: float, yPos: float) -> None:
        """Adds a new node to a Network instance"""
        thisNode = Node(nodeID, xPos, yPos)
        self.nodesDict[nodeID] = thisNode

    def addEdgeToDict(self, edgeID: tuple, index: int) -> None:
        """Adds a new edge to the dict with the value of its index in the edges array"""
        self.edgesDict[edgeID] = index

    def addArcToDict(self, numID: int, arcID: tuple, distance: float, FC: float, VC: float) -> None:
        """Adds a new arc to a Network instance"""
        thisArc = Arc(numID, (arcID[0], arcID[1]), arcID[2], distance, FC, VC)
        self.arcsDict[arcID] = thisArc

    # ===================================================================
    # ============== EDGE/ARC GETTER/SETTER/HELPER METHODS ==============
    # ===================================================================
    def getNumEdges(self) -> int:
        """Returns the number of edges in the network"""
        return self.numEdges

    def getEdgesArray(self) -> ndarray:
        """Returns the NumPy array of the edge ID tuples"""
        return self.edgesArray

    def getDistancesArray(self) -> ndarray:
        """Returns the NumPy array of the edge distances"""
        return self.distancesArray

    def getNumArcCaps(self) -> int:
        """Returns the number of possible arc capacities for the network"""
        return self.numArcCaps

    def getPossibleArcCaps(self) -> ndarray:
        """Returns the possible arc capacities for the network"""
        return self.possibleArcCapsArray

    def getArcObject(self, arcID: tuple) -> Arc:
        """Returns the arc object with the given ID"""
        return self.arcsDict[arcID]

    def getArcEdge(self, arcID: tuple) -> tuple:
        """Returns the edge that the arc spans"""
        arc = self.arcsDict[arcID]
        return arc.edgeID

    def getArcCapacity(self, arcID: tuple) -> int:
        """Returns the arc's capacity"""
        arc = self.arcsDict[arcID]
        return arc.capacity

    def getArcDistance(self, arcID: tuple) -> float:
        """Returns the arc's distance"""
        arc = self.arcsDict[arcID]
        return arc.distance

    def getArcFixedCost(self, arcID: tuple) -> float:
        """Returns the arc's fixed cost"""
        arc = self.arcsDict[arcID]
        return arc.fixedCost

    def getArcVariableCost(self, arcID: tuple) -> float:
        """Returns the arc's variable cost"""
        arc = self.arcsDict[arcID]
        return arc.variableCost

    def getArcFixedCostFromEdgeCapIndices(self, edgeIndex: int, capIndex: int) -> float:
        """Gets the arc fixed cost from the edge index and the capacity index"""
        arcFixedCost = self.arcsMatrix[self.arcsDict[(self.edgesArray[edgeIndex][0], self.edgesArray[edgeIndex][1],
                                                      self.possibleArcCapsArray[capIndex])].numID][5]
        return arcFixedCost

    def getArcVariableCostFromEdgeCapIndices(self, edgeIndex: int, capIndex: int) -> float:
        """Gets the arc variable cost from the edge index and the capacity index"""
        arcVariableCost = self.arcsMatrix[self.arcsDict[(self.edgesArray[edgeIndex][0], self.edgesArray[edgeIndex][1],
                                                         self.possibleArcCapsArray[capIndex])].numID][6]
        return arcVariableCost

    # ===============================================================
    # ============== NODE GETTER/SETTER/HELPER METHODS ==============
    # ===============================================================
    def getNumTotalNodes(self) -> int:
        """Returns the total number of nodes in the network"""
        return self.numTotalNodes

    def getNodeObject(self, nodeID: int) -> Node:
        """Returns the node object with the node ID"""
        return self.nodesDict[nodeID]

    def getNodeCoordinates(self, nodeID: int) -> tuple:
        """Returns the (x,y) tuple position of the node"""
        node = self.nodesDict[nodeID]
        return node.xPos, node.yPos

    def getNodeType(self, nodeID: int) -> int:
        """Returns the type of the node (NOTE: nodeType = {0: source, 1: sink, 2: intermediate})"""
        node = self.nodesDict[nodeID]
        return node.nodeType

    def getNodeIncomingEdges(self, nodeID: int) -> ndarray:
        """Returns the incoming edges of the node"""
        node = self.nodesDict[nodeID]
        return node.incomingEdges

    def getNodeOutgoingEdges(self, nodeID: int) -> ndarray:
        """Returns the outgoing edges of the node"""
        node = self.nodesDict[nodeID]
        return node.outgoingEdges

    def getNumSources(self) -> int:
        """Returns the number of sources in the network"""
        return self.numSources

    def getNumSinks(self) -> int:
        """Returns the number of sinks in the network"""
        return self.numSinks

    def getNumInterNodes(self) -> int:
        """Returns the number of intermediate nodes in the network"""
        return self.numInterNodes

    def getSourcesArray(self) -> ndarray:
        """Returns the NumPy sources array for the network"""
        return self.sourcesArray

    def getSinksArray(self) -> ndarray:
        """Returns the NumPy sinks array for the network"""
        return self.sinksArray

    def getInterArray(self) -> ndarray:
        """Returns the NumPy intermediate array for the network"""
        return self.sinksArray

    def setNodeType(self, nodeID: int, nodeType: int) -> None:
        """Sets a node to a type (NOTE: nodeType = {0: source, 1: sink, 2: intermediate})"""
        thisNode = self.nodesDict[nodeID]
        thisNode.nodeType = nodeType

    def addIncomingEdgeToNode(self, nodeID: int, incomingEdge: tuple) -> None:
        """Adds the edge to the node's incoming edge list"""
        thisNode = self.nodesDict[nodeID]
        thisNode.addIncomingEdge(incomingEdge)

    def addOutgoingEdgeToNode(self, nodeID: int, outgoingEdge: tuple) -> None:
        """Adds the edge to the node's outgoing edge list"""
        thisNode = self.nodesDict[nodeID]
        thisNode.addOutgoingEdge(outgoingEdge)

    # ===========================================
    # ============== PRINT METHODS ==============
    # ===========================================
    def printAllNodeData(self) -> None:
        """Prints all the data for each node in the network"""
        for node in self.nodesDict.values():
            node.printNodeData()

    def printAllArcData(self) -> None:
        """Prints all the data for each arc in the network"""
        for arc in self.arcsDict.values():
            arc.printArcData()
