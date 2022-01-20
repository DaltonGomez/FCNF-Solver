class Node:
    """Class that defines an edge in a Fixed Charge Flow Network."""

    def __init__(self, edgeID: str, fromNode: str, toNode: str, fixedCost: int, variableCost: int, capacity: int):
        """Constructor of an edge instance."""
        # Input network attributes
        self.edgeID = edgeID
        self.fromNode = fromNode
        self.toNode = toNode
        self.fixedCost = fixedCost
        self.variableCost = variableCost
        self.capacity = capacity

        # Solution attributes
        self.opened = False
        self.flow = 0
        self.totalCost = 0
