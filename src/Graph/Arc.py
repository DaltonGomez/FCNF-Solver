from typing import Tuple


class Arc:
    """Class that defines an arc in a Candidate Graph object"""

    def __init__(self, numID: int, edgeID: tuple, capacity: int, distance: float, fixedCost: float,
                 variableCost: float):
        """Constructor of an arc instance.
        Edges are unidirectional connections between two nodes.
        Several unidirectional arcs may span the same edge, each with their own capacity."""
        # Arc attributes
        self.numID: int = numID  # Unique identifier of the arc
        self.edgeID: Tuple[int, int] = edgeID  # Tuple representing the arc the edge spans, given as (fromNodeID, toNodeID)
        self.capacity = capacity  # Max flow that can be assigned to the arc
        self.distance = distance  # Euclidean distance of the arc, assigned at graph generation
        self.fixedCost = fixedCost  # Fixed cost of opening the arc (i.e. paid in full if the arc's flow is greater than zero)
        self.variableCost = variableCost  # Variable cost of using the arc per unit of flow assigned (i.e. paid out as assignedFlow * variableCost)

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
