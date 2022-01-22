class Edge:
    """Class that defines an edge in a Fixed Charge Network Flow w/ Parallel Edges"""

    def __init__(self, edgeID: str, fromNode: str, toNode: str):
        """Constructor of an edge instance"""
        # Input network attributes
        self.edgeID = edgeID
        self.fromNode = fromNode
        self.toNode = toNode

        # Solution attributes - NOTE: The capacity, FC, and VC are determined when an edge is selected by the solver
        self.opened = False
        self.fixedCost = 0
        self.variableCost = 0
        self.capacity = 0
        self.flow = 0
        self.totalCost = 0

    def printEdgeData(self):
        """Prints all relevant data for an edge"""
        print("=============== Edge ===============")
        print("Edge ID = " + self.edgeID)
        print("From-Node = " + self.fromNode)
        print("To-Node = " + self.toNode)
        print("\nOpened = " + str(self.opened))
        print("Fixed Cost = " + str(self.fixedCost))
        print("Variable Cost = " + str(self.variableCost))
        print("Capacity = " + str(self.capacity))
        print("Flow = " + str(self.flow))
        print("Total Cost = " + str(self.totalCost))
