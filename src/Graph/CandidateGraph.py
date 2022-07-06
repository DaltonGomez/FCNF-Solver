import os
import pickle
from typing import Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.spatial import Delaunay

from src.Graph.Arc import Arc
from src.Graph.Node import Node


class CandidateGraph:
    """Class that defines a Candidate Graph object with multi-source/multi-sink and parallel edges, called arcs"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self):
        """Constructor of a Candidate Graph instance"""
        self.name: str = ""  # Graph name, assigned at graph generation
        # Node Attributes
        self.numTotalNodes: int = 0  # Total number of nodes in the graph
        self.nodesDict: Dict[int, Node] = {}  # Dictionary mapping nodeID keys to values of Node objects
        self.numSources: int = 0  # Total number of sources in the graph
        self.sourcesArray: ndarray = np.array(0, dtype='i')  # Numpy array containing the nodeID of all sources
        self.numSinks: int = 0  # Total number of sinks in the graph
        self.sinksArray: ndarray = np.array(0, dtype='i')  # Numpy array containing the nodeID of all sinks
        self.numInterNodes: int = 0  # Total number of intermediate nodes in the graph
        self.interNodesArray: ndarray = np.array(0, dtype='i')  # Numpy array containing the nodeID of all intermediate nodes
        # Edge Attributes
        self.numEdges: int = 0  # Total number of edges nodes in the graph
        self.edgesArray: ndarray = np.array(0, dtype='i')  # Numpy array containing the edgesID as (fromNode, toNode) of all edges
        self.edgesDict: Dict[Tuple[int, int]] = {}  # Dictionary mapping edgeID keys, given as (fromNode, toNode), to values of edge index in the edgesArray attribute values
        # Arc Attributes
        self.numArcsPerEdge: int = 0  # Total number of possible arc capacities in the graph (i.e. number of parallel arcs per edge)
        self.possibleArcCapsArray: ndarray = np.array(0, dtype='f')  # Numpy array containing possible arc capacities
        self.numTotalArcs: int = 0  # Total number of arcs in the entire graph
        self.arcsDict: Dict[Tuple[int, int, float], Arc] = {}  # Dictionary mapping arcID keys, given as (fromNode, toNode, capacity), to values of Arc objects
        self.arcsMatrix: ndarray = np.array(0, dtype='f')  # Numpy matrix with seven columns, representing: [arcNumID, fromNode, toNode, capacity, distance, fixedCost, variableCost]
        # Capacitated/Charged Source/Sink Generalization
        self.isSourceSinkCapacitated: bool = True  # Boolean flag indicating if sources and sinks are capacitated
        self.sourceCapsArray: ndarray = np.array(0, dtype='f')  # Numpy array containing the capacity of each source, where the index matches the sources array
        self.sinkCapsArray: ndarray = np.array(0, dtype='f')  # Numpy array containing the capacity of each sink, where the index matches the sink array
        self.totalPossibleDemand: float = 0.0  # Maximum possible demand of the candidate graph as determined by the minimum of all the source capacities or sink capacities
        # Variable Cost Source/Sink Generalization
        self.isSourceSinkCharged: bool = False  # Boolean flag indicating if sources and sinks are charged with a variable cost
        self.sourceVariableCostsArray: ndarray = np.array(0, dtype='f')  # Numpy array containing the variable cost of each source, where the index matches the sources array
        self.sinkVariableCostsArray: ndarray = np.array(0, dtype='f')  # Numpy array containing the variable cost of each sink, where the index matches the sinks array
        # Visualization Attributes
        self.points: ndarray = np.array(0, dtype='f')  # Numpy array containing the x-position and y-position of each node, where the index matches the nodeID (used for drawing the triangulation)
        self.distancesArray: ndarray = np.array(0, dtype='f')  # Numpy array containing the distances of each edge, where the index matches the edges array

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def drawGraphTriangulation(self) -> None:
        """Draws the Delaunay triangulation of the candidate graph with MatPlotLib for quick judgement of topology"""
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
        graphFile = self.name + ".png"
        catPath = os.path.join(currDir, "data/graph_instances/delaunay_figs", graphFile)
        plt.savefig(catPath)
        plt.show()

    # =================================================
    # ============== DATA IN/OUT METHODS ==============
    # =================================================
    def saveCandidateGraph(self) -> None:
        """Saves the candidate graph instance to disc via a pickle dump"""
        # Path management
        currDir = os.getcwd()
        graphFile = self.name + ".p"
        catPath = os.path.join(currDir, "data/graph_instances", graphFile)
        print("Saving " + graphFile + " to: " + catPath)
        # Pickle dump
        pickle.dump(self, open(catPath, "wb"))

    @staticmethod
    def loadCandidateGraph(graphFile: str):
        """Loads a candidate graph instance via a pickle load"""
        # Path management
        currDir = os.getcwd()
        catPath = os.path.join(currDir, "data/graph_instances", graphFile)
        print("Loading " + graphFile + " from: " + catPath)
        # Pickle load
        candidateGraph = pickle.load(open(catPath, "rb"))
        return candidateGraph

    # ============================================
    # ============== HELPER METHODS ==============
    # ============================================
    def addNodeToDict(self, nodeID: int, xPos: float, yPos: float) -> None:
        """Adds a new node to a candidate graph instance"""
        thisNode = Node(nodeID, xPos, yPos)
        self.nodesDict[nodeID] = thisNode

    def addEdgeToDict(self, edgeID: tuple, index: int) -> None:
        """Adds a new edge to the dict with the value of its index in the edges array"""
        self.edgesDict[edgeID] = index

    def addArcToDict(self, numID: int, arcID: tuple, distance: float, FC: float, VC: float) -> None:
        """Adds a new arc to a candidate graph instance"""
        thisArc = Arc(numID, (arcID[0], arcID[1]), arcID[2], distance, FC, VC)
        self.arcsDict[arcID] = thisArc

    def addIncomingEdgeToNode(self, nodeID: int, incomingEdge: tuple) -> None:
        """Adds the edge to the node's incoming edge list"""
        thisNode = self.nodesDict[nodeID]
        thisNode.addIncomingEdge(incomingEdge)

    def addOutgoingEdgeToNode(self, nodeID: int, outgoingEdge: tuple) -> None:
        """Adds the edge to the node's outgoing edge list"""
        thisNode = self.nodesDict[nodeID]
        thisNode.addOutgoingEdge(outgoingEdge)

    def calculateTotalPossibleDemand(self) -> float:
        """Calculates the total possible demand as the minimum of the total source or total sink capacity"""
        totalSrcCapacity = 0.0
        for srcCap in self.sourceCapsArray:
            totalSrcCapacity += srcCap
        totalSinkCapacity = 0.0
        for sinkCap in self.sinkCapsArray:
            totalSinkCapacity += sinkCap
        return min(totalSrcCapacity, totalSinkCapacity)

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

    def getNodeCoordinates(self, nodeID: int) -> tuple:
        """Returns the (x,y) tuple position of the node"""
        node = self.nodesDict[nodeID]
        return node.xPos, node.yPos

    def setNodeType(self, nodeID: int, nodeType: int) -> None:
        """Sets a node to a type (NOTE: nodeType = {0: source, 1: sink, 2: intermediate})"""
        thisNode = self.nodesDict[nodeID]
        thisNode.nodeType = nodeType

    # ===========================================
    # ============== PRINT METHODS ==============
    # ===========================================
    def printAllNodeData(self) -> None:
        """Prints all the data for each node in the candidate graph"""
        for node in self.nodesDict.values():
            node.printNodeData()

    def printAllArcData(self) -> None:
        """Prints all the data for each arc in the candidate graph"""
        for arc in self.arcsDict.values():
            arc.printArcData()
