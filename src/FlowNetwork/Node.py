import numpy as np
from numpy import ndarray


class Node:
    """Class that defines a node in a Network object"""

    def __init__(self, nodeID: int, nodeType: int, incomingEdges: ndarray, outgoingEdges: ndarray):
        """Constructor of a node instance (NOTE: nodeType = {0: source, 1: sink, 2: intermediate})"""
        # Input network attributes
        self.nodeID = nodeID
        self.nodeType = nodeType
        # Topology attributes
        self.incomingEdges = incomingEdges
        self.outgoingEdges = outgoingEdges

    def getNodeID(self) -> int:
        """Returns the node ID of the node object"""
        return self.nodeID

    def getNodeType(self) -> int:
        """Returns the node type of the node object"""
        return self.nodeType

    def setNodeType(self, nodeType: int) -> None:
        """Reassigns the node type of the node object (NOTE: nodeType = {0: source, 1: sink, 2: intermediate})"""
        self.nodeType = nodeType

    def getIncomingEdges(self) -> ndarray:
        """Returns the incoming edges of the node object"""
        return self.incomingEdges

    def setIncomingEdges(self, incomingEdges: ndarray) -> None:
        """Reassigns the incoming edges of the node object"""
        self.incomingEdges = incomingEdges

    def addIncomingEdges(self, edgesToAdd: ndarray) -> None:
        """Concatenates the new edges onto the current incoming edges"""
        self.incomingEdges = np.concatenate((self.incomingEdges, edgesToAdd))

    def getOutgoingEdges(self) -> ndarray:
        """Returns the outgoing edges of the node object"""
        return self.outgoingEdges

    def setOutgoingEdges(self, outgoingEdges: ndarray) -> None:
        """Reassigns the incoming edges of the node object"""
        self.outgoingEdges = outgoingEdges

    def addOutgoingEdges(self, edgesToAdd: ndarray) -> None:
        """Concatenates the new edges onto the current outgoing edges"""
        self.outgoingEdges = np.concatenate((self.outgoingEdges, edgesToAdd))

    def printNodeData(self) -> None:
        """Prints all relevant data for a node"""
        print("=============== NODE ===============")
        print("Node ID = " + str(self.nodeID))
        print("Node Type = " + str(self.nodeType))
        print("NOTE: nodeType = {0: source, 1: sink, 2: intermediate}")
        print("Incoming Edges = " + str(self.incomingEdges))
        print("Outgoing Edges = " + str(self.outgoingEdges))
