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
        """Constructor of a FCNF instance"""
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
        self.distancesArray = None
        self.numArcCaps = 0
        self.possibleArcCaps = None
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

    def addNodeToDict(self, nodeID: int, xPos: float, yPos: float) -> None:
        """Adds a new node to a Network instance"""
        thisNode = Node(nodeID, xPos, yPos)
        self.nodesDict[nodeID] = thisNode

    def addArcToDict(self, arcID: tuple, distance: float, FC: float, VC: float) -> None:
        """Adds a new arc to a Network instance"""
        thisArc = Arc((arcID[0], arcID[1]), arcID[2], distance, FC, VC)
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
        return self.possibleArcCaps

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
        npIncomingEdge = np.array(incomingEdge)
        thisNode.addIncomingEdge(npIncomingEdge)

    def addOutgoingEdgeToNode(self, nodeID: int, outgoingEdge: tuple) -> None:
        """Adds the edge to the node's outgoing edge list"""
        thisNode = self.nodesDict[nodeID]
        npOutgoingEdge = np.array(outgoingEdge)
        thisNode.addIncomingEdge(npOutgoingEdge)

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def drawNetworkTriangulation(self):
        """Draws the Delaunay triangulation of the network with MatPlotLib for quick judgement of topology"""
        triangulation = Delaunay(self.points)
        plt.triplot(self.points[:, 0], self.points[:, 1], triangulation.simplices)
        plt.plot(self.points[:, 0], self.points[:, 1], 'ko')
        sourceList = []
        for source in self.sourcesArray:
            thisPoint = self.getNodeCoordinates(source)
            sourceList.append(thisPoint)
        sourcePoints = np.array(sourceList)
        plt.plot(sourcePoints[:, 0], sourcePoints[:, 1], 'bD')
        sinkList = []
        for sink in self.sinksArray:
            thisPoint = self.getNodeCoordinates(sink)
            sinkList.append(thisPoint)
        sinkPoints = np.array(sinkList)
        plt.plot(sinkPoints[:, 0], sinkPoints[:, 1], 'rs')
        plt.show()

    # =================================================
    # ============== DATA IN/OUT METHODS ==============
    # =================================================
    def saveNetwork(self) -> None:
        """Saves the network to disc via a pickle dump"""
        pickle.dump(self, open(self.name + ".p", "wb"))

    @staticmethod
    def loadNetwork(name: str):
        """Loads a FCFN from a text file encoding"""
        flowNetwork = pickle.load(open(name + ".p", "rb"))
        return flowNetwork

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
