from typing import List, Tuple


class Node:
    """Class that defines a node in a Candidate Graph object"""

    def __init__(self, nodeID: int, xPos: float, yPos: float):
        """Constructor of a node instance"""
        self.nodeID: int = nodeID  # Unique identifier of the node
        self.nodeType: int = 2  # Type of the node, encoded from nodeType = {0: source, 1: sink, 2: intermediate}
        self.xPos: float = xPos  # Horizontal position in the 2D embedding
        self.yPos: float = yPos  # Vertical position in the 2D embedding
        self.incomingEdges: List[Tuple[int, int]] = []  # List of incoming edges, where each edge is given as a tuple of (fromNodeID, toNodeID) and self.nodeID == toNodeID
        self.outgoingEdges: List[Tuple[int, int]] = []  # List of outgoing edges, where each edge is given as a tuple of (fromNodeID, toNodeID) and self.nodeID == fromNodeID

    def addIncomingEdge(self, edgeToAdd: Tuple[int, int]) -> None:
        """Concatenates the new edge onto the current incoming edges"""
        self.incomingEdges.append(edgeToAdd)

    def addOutgoingEdge(self, edgeToAdd: Tuple[int, int]) -> None:
        """Concatenates the new edge onto the current outgoing edges"""
        self.outgoingEdges.append(edgeToAdd)

    def printNodeData(self) -> None:
        """Prints all relevant data for a node"""
        print("=============== NODE ===============")
        print("Node ID = " + str(self.nodeID))
        print("Position = (" + str(self.xPos) + ", " + str(self.xPos) + ")")
        print("Node Type = " + str(self.nodeType))
        print("NOTE: nodeType = {0: source, 1: sink, 2: intermediate}")
        print("Incoming Edges = " + str(self.incomingEdges))
        print("Outgoing Edges = " + str(self.outgoingEdges))
