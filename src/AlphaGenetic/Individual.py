
from typing import Dict, Tuple, List

from numpy import ndarray

from src.FlowNetwork.CandidateGraph import CandidateGraph
from src.FlowNetwork.FlowNetworkSolution import FlowNetworkSolution


class Individual:
    """Class that defines an individual in the GA population"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, ID: int, graph: CandidateGraph, initialAlphaValues: ndarray):
        """Constructor of an Individual instance"""
        # Input candidate graph and ID in the population
        self.id: int = ID  # Integer ID specifying the individuals index in the population
        self.graph: CandidateGraph = graph  # Input candidate graph instance
        # Alpha Values (a.k.a. the genotype of the individual)
        self.alphaValues: ndarray = initialAlphaValues  # Numpy array of alpha values assigned to the individual at initialization
        # Expressed FlowNetwork (a.k.a. the phenotype of the individual)
        self.isSolved: bool = False  # Flip true when a relaxed-LP solver runs/returns solution data; flip false when the alpha values array is modified
        self.arcFlows: Dict[Tuple[int, int], float] = {}  # Dictionary mapping (edgeIndex, capIndex) keys to values of assigned flow
        self.srcFlows: List[float] = []  # List of assigned flows on each source, where indices match graph.sourcesArray
        self.sinkFlows: List[float] = []  # List of assigned flows on each sink, where indices match graph.sourcesArray
        # Returned Cost (a.k.a. the fitness of the individual)
        self.trueCost: float = 0.0  # True cost computed by using the assigned flows and the Fixed Charge Network Flow model formulation
        self.fakeCost: float = 0.0  # Actual objective value returned by the solver (used only for debugging purposes)

    # ============================================
    # ============== HELPER METHODS ==============
    # ============================================
    def setAlphaValues(self, alphaValues: ndarray) -> None:
        """Resets the alpha values to a new array"""
        self.alphaValues = alphaValues
        self.isSolved = False  # Any change to alpha values should reset the isSolved bool

    def resetOutputNetwork(self) -> None:
        """Resets the expressed network (i.e. phenotype) output data in the individual"""
        self.isSolved = False
        self.srcFlows = []
        self.sinkFlows = []
        self.arcFlows = {}
        self.trueCost = 0.0
        self.fakeCost = 0.0

    def resetCostValues(self) -> None:
        """Resets just the cost values (i.e. fitness) of the individual"""
        self.isSolved = False
        self.trueCost = 0.0
        self.fakeCost = 0.0

    def writeIndividualAsSolution(self, minTargetFlow: float, optionalDescription="") -> FlowNetworkSolution:
        """Writes the current expressed network as a solution instance"""
        if self.isSolved is False:
            print("You must solve the individual before writing a solution!")
        else:
            print("Writing solution from individual #" + str(self.id) + "...")
            thisSolution = FlowNetworkSolution(self.graph, minTargetFlow, self.fakeCost, self.trueCost,
                                               self.srcFlows, self.sinkFlows, self.arcFlows, "Alpha_GA", False,
                                               self.graph.isSourceSinkCapacitated, self.graph.isSourceSinkCharged,
                                               optionalDescription=optionalDescription)
            return thisSolution
