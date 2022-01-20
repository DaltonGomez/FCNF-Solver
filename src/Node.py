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
            self.nodeType = "transshipment"
        self.variableCost = variableCost
        self.capacity = capacity
        # TODO- Add in fixed costs to source and sink nodes if desired

        # Solution attributes
        self.opened = False
        self.inFlow = 0
        self.outFlow = 0
        self.totalCost = 0

    def printNodeData(self):
        """Prints all relevant data for a node"""
        print("=============== NODE ===============")
        print("Node ID = " + self.nodeID)
        print("Node Type = " + self.nodeType)
        print("Variable Cost = " + str(self.variableCost))
        print("Capacity = " + str(self.capacity))
        print("\nOpened = " + str(self.opened))
        print("In-Flow = " + str(self.inFlow))
        print("Out-Flow = " + str(self.outFlow))
        print("Total Cost = " + str(self.totalCost))
