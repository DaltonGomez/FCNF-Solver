class Arc:
    """Class that defines an arc in a Network object"""

    def __init__(self, edgeID: tuple, capacity: int, distance: float, fixedCost: float, variableCost: float):
        """Constructor of an edge instance"""
        # Input network attributes
        self.edgeID = edgeID
        self.capacity = capacity
        self.distance = distance
        self.fixedCost = fixedCost
        self.variableCost = variableCost

    def getArcID(self) -> tuple:
        """Returns the ID of an arc as the tuple (fromNode, toNode, capacity)"""
        return self.edgeID[0], self.edgeID[1], self.capacity

    def getEdgeID(self) -> tuple:
        """Returns the ID of an edge as the tuple (fromNode, toNode)"""
        return self.edgeID

    def getCapacity(self) -> int:
        """Returns the capacity of the arc"""
        return self.capacity

    def getDistance(self) -> float:
        """Returns the Euclidean distance of the edge"""
        return self.distance

    def getFixedCost(self) -> float:
        """Returns the fixed cost of the edge"""
        return self.fixedCost

    def getVariableCost(self) -> float:
        """Returns the variable cost of the edge"""
        return self.variableCost

    def printArcData(self) -> None:
        """Prints all data for an arc"""
        print("=============== Arc ===============")
        print("Arc ID = " + str((self.edgeID[0], self.edgeID[1], self.capacity)))
        print("From-Node = " + str(self.edgeID[0]))
        print("To-Node = " + str(self.edgeID[1]))
        print("Capacity = " + str(self.capacity))
        print("Distance = " + str(round(self.distance, 2)))
        print("Fixed Cost = " + str(round(self.fixedCost, 2)))
        print("Variable Cost = " + str(round(self.variableCost, 2)))
