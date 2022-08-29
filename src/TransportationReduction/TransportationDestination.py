
from Graph.Node import Node


class TransportationDestination:
    """Class that defines a Destination object in the FCTP reduction, representing nodes in the original input graph"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, originalNode: Node):
        """Constructor of a Transportation Node instance"""
        self.id: int = originalNode.nodeID
        self.originalArc: Node = originalNode
        self.demand: float = 0.0  # TODO - Update per the rules of the reduction
