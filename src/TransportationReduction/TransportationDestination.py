from typing import Tuple

from Graph.Node import Node


class TransportationDestination:
    """Class that defines a Destination object in the FCTP reduction, representing nodes in the original input graph"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, originalNode: Node, demand: float):
        """Constructor of a Transportation Node instance"""
        self.destinationID: int = originalNode.nodeID
        self.originalNode: Node = originalNode
        self.demand: float = demand
        self.incomingTransportArcs: list = []

    def addIncomingTransportArc(self, incomingTransportArc: Tuple[Tuple[int, int, float], int]):
        """Adds an incoming transport arc to the destination node, formatted as ((fromNode, toNode, cap), destination)"""
        self.incomingTransportArcs.append(incomingTransportArc)
