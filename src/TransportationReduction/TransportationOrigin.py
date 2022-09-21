from typing import Tuple

from Graph.Arc import Arc


class TransportationOrigin:
    """Class that defines an Origin object in the FCTP reduction, representing arcs in the original input graph"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, originalArc: Arc):
        """Constructor of a Transportation Origin instance"""
        self.originID: Tuple[int, int, float] = (originalArc.edgeID[0], originalArc.edgeID[1], originalArc.capacity)
        self.originalArc: Arc = originalArc
        self.supply: float = originalArc.capacity
