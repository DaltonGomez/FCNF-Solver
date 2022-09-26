from typing import List, Tuple


class Pathlet:
    """Class that stores the information for a pathlet in the flow network solution to a candidate graph instance"""

    def __init__(self, startingNodeID: int):
        """Constructor of a pathlet instance"""
        # Composition attributes
        self.nodes: List[int] = [startingNodeID]  # List of all node IDs that make up the pathlet
        self.arcs: List[Tuple[int, int]] = []  # List of all arc indexes that make up the pathlet
        # Descriptive attributes
        self.size: int = -1  # Number of arcs in the pathlet
        self.totalFlow: float = -1.0  # Amount of flow transported by pathlet
        self.variableCost: float = -1.0  # Variable cost of the pathlet
        self.fixedCost: float = -1.0  # Fixed cost of the pathlet
        self.totalCost: float = -1.0  # Cost of the edges in the pathlet
        self.totalCostPerFlow: float = -1.0  # Ratio of flow transported to total cost of the pathlet
