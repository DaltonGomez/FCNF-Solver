class Node:
    """Class that defines a node in a Fixed Charge Flow Network"""

    def __init__(self, nodeID: str, variableCost: int, capacity: int):
        """Constructor of a node instance"""
        # Input network attributes
        self.nodeID = nodeID
        if nodeID[0] == "s":
            self.nodeType = "source"
        elif nodeID[0] == "t":
            self.nodeType = "sink"
        elif nodeID[0] == "n":
            self.nodeType = "intermediate"
        self.variableCost = variableCost
        self.capacity = capacity

        # Topology attributes
        self.incomingEdges = []
        self.outgoingEdges = []

        # Solution attributes
        self.opened = False
        self.flow = 0
        self.totalCost = 0

    def printNodeData(self) -> None:
        """Prints all relevant data for a node"""
        print("=============== NODE ===============")
        print("Node ID = " + self.nodeID)
        print("Node Type = " + self.nodeType)
        print("Variable Cost = " + str(self.variableCost))
        print("Capacity = " + str(self.capacity))
        print("\nIncoming Edges = " + str(self.incomingEdges))
        print("Outgoing Edges = " + str(self.outgoingEdges))
        print("\nOpened = " + str(self.opened))
        print("Flow = " + str(self.flow))
        print("Total Cost = " + str(self.totalCost))
