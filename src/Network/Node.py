class Node:
    """Class that defines a node in a Network object"""

    def __init__(self, nodeID: int, xPos: float, yPos: float):
        """Constructor of a node instance (NOTE: nodeType = {0: source, 1: sink, 2: intermediate})"""
        self.nodeID = nodeID
        self.nodeType = 2
        self.xPos = xPos
        self.yPos = yPos
        self.incomingEdges = []
        self.outgoingEdges = []

    def getNodeID(self) -> int:
        """Returns the node ID of the node object"""
        return self.nodeID

    def getNodeType(self) -> int:
        """Returns the node type of the node object"""
        return self.nodeType

    def setNodeType(self, nodeType: int) -> None:
        """Reassigns the node type of the node object (NOTE: nodeType = {0: source, 1: sink, 2: intermediate})"""
        self.nodeType = nodeType

    def getXPos(self) -> float:
        """Returns the x-position of the node"""
        return self.xPos

    def getYPos(self) -> float:
        """Returns the y-position of the node"""
        return self.yPos

    def getIncomingEdges(self) -> list:
        """Returns the incoming edges of the node object"""
        return self.incomingEdges

    def setIncomingEdges(self, incomingEdges: list) -> None:
        """Reassigns the incoming edges of the node object"""
        self.incomingEdges = incomingEdges

    def addIncomingEdge(self, edgeToAdd: tuple) -> None:
        """Concatenates the new edge onto the current incoming edges"""
        self.incomingEdges.append(edgeToAdd)

    def getOutgoingEdges(self) -> list:
        """Returns the outgoing edges of the node object"""
        return self.outgoingEdges

    def setOutgoingEdges(self, outgoingEdges: list) -> None:
        """Reassigns the incoming edges of the node object"""
        self.outgoingEdges = outgoingEdges

    def addOutgoingEdge(self, edgeToAdd: tuple) -> None:
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
