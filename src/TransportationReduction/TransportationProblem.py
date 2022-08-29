from typing import Dict

from Graph.CandidateGraph import CandidateGraph


class TransportationProblem:
    """Class that defines a Transportation Problem object, instantiated by reducing a Candidate Graph"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, graphToReduce: CandidateGraph):
        """Constructor of a Transportation Problem instance"""
        self.origins: Dict = {}
        self.destinations: Dict = {}
        self.arcs: Dict = {}
        self.performReduction(graphToReduce)
        # TODO - Decide if the input graph to reduce should be held as an instance attribute

    # ===============================================
    # ============== REDUCTION METHODS ==============
    # ===============================================
    def performReduction(self, inputGraph: CandidateGraph) -> None:
        """Performs the reduction to the Fixed-Charge Transportation Problem on the input candidate graph instance"""

