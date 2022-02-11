class Edge:
    """Class that defines an edge in a Fixed Charge Flow Network (without Parallel Edges)"""

    def __init__(self, edgeID: str, fromNode: str, toNode: str, fixedCost: int, variableCost: int, capacity: int):
        """Constructor of an edge instance"""
        # Input network attributes
        self.edgeID = edgeID
        self.fromNode = fromNode
        self.toNode = toNode
        self.capacity = capacity
        self.fixedCost = fixedCost
        self.variableCost = variableCost

        # Solution attributes
        self.opened = False
        self.flow = 0
        self.totalCost = 0

    def printEdgeData(self):
        """Prints all relevant data for an edge"""
        print("=============== Edge ===============")
        print("Edge ID = " + self.edgeID)
        print("From-Node = " + self.fromNode)
        print("To-Node = " + self.toNode)
        print("Fixed Cost = " + str(self.fixedCost))
        print("Variable Cost = " + str(self.variableCost))
        print("Capacity = " + str(self.capacity))
        print("\nOpened = " + str(self.opened))
        print("Flow = " + str(self.flow))
        print("Total Cost = " + str(self.totalCost))
