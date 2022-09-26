from typing import List


class Pathlet:
    """Class that stores the information for a pathlet in the flow network solution to a candidate graph instance"""

    def __init__(self, startingNode: int):
        """Constructor of a pathlet instance"""
        # Composition attributes
        self.startingNode: int = startingNode  # Integer ID of the node that begins the pathlet
        self.endingNode: int = -1  # Integer ID of the node that ends the pathlet
        self.allNodes: List = []  # List of all node IDs that make up the pathlet
        # Descriptive attributes
        self.size: int = -1  # Number of edges in the pathlet
        self.totalFlow: float = -1.0  # Amount of flow transported by pathlet
        self.variableCost: float = -1.0
        self.fixedCost: float = -1.0
        self.totalCost: float = -1.0  # Cost of the edges in the pathlet
        self.totalCostPerFlow: float = -1.0
