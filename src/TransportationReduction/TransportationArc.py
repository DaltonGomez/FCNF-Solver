from typing import Tuple


class TransportationArc:
    """Class that defines an Arc object in the FCTP reduction"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, transportArcID: Tuple[Tuple[int, int, float], int], fixedCost: float, variableCost: float):
        """Constructor of a Transportation Arc instance"""
        self.transportArcID: Tuple[Tuple[int, int, float], int] = transportArcID
        self.fixedCost: float = fixedCost
        self.variableCost: float = variableCost
