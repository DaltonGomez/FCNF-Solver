
from typing import Dict, Tuple, List

from numpy import ndarray

from FlowNetwork.Pathlet import Pathlet
from src.FlowNetwork.FlowNetworkSolution import FlowNetworkSolution
from src.Graph.CandidateGraph import CandidateGraph


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
        # Pathlet Attributes
        self.pathlets: List[Pathlet] = []  # Holds the set of pathlets that make up the flow network solution

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
            print("ERROR - Cannot write a solution from an unsolved individual!")
        else:
            thisSolution = FlowNetworkSolution(self.graph, minTargetFlow, self.fakeCost, self.trueCost,
                                               self.srcFlows, self.sinkFlows, self.arcFlows, "Alpha_GA", False,
                                               self.graph.isSourceSinkCapacitated, self.graph.isSourceSinkCharged,
                                               optionalDescription=optionalDescription)
            return thisSolution

    # =============================================
    # ============== PATHLET METHODS ==============
    # =============================================
    def computeAllPathlets(self) -> None:
        """Computes all the pathlets that have a positive flow"""
        # Declare a stack to maintain which nodes still need to be checked
        uncheckedNodesStack = []
        # Iterate over all sources and push them on the stack if they have a positive flow
        for srcIndex in range(self.graph.numSources):
            if self.srcFlows[srcIndex] > 0:
                nodeID = self.graph.sourcesArray[srcIndex]
                uncheckedNodesStack.append(nodeID)
        # Until the stack is empty, evaluate each node and build pathlets
        while len(uncheckedNodesStack) > 0:
            thisStartingNode = uncheckedNodesStack.pop()
            self.evaluateStartingNode(thisStartingNode, uncheckedNodesStack)

    def evaluateStartingNode(self, startingNodeID: int, uncheckedNodeStack: List[int]) -> None:
        """Builds all possible pathlets that start at the starting node and appends them to the individual's pathlets"""
        currentNodeObj = self.graph.nodesDict[startingNodeID]
        for outgoingEdge in currentNodeObj.outgoingEdges:
            edgeIndex = self.graph.edgesDict[outgoingEdge]
            for capIndex in range(len(self.graph.possibleArcCapsArray)):
                capacity = self.graph.possibleArcCapsArray[capIndex]
                arcObj = self.graph.arcsDict[(outgoingEdge[0], outgoingEdge[1], capacity)]
                if self.arcFlows[(edgeIndex, capIndex)] > 0.0:
                    pass
                    # TODO - Implement the build pathlet method and call here on each outgoing flow

    def buildPathlet(self, startingNodeID: int, uncheckedNodeStack: List[int]) -> Pathlet:
        """Builds the pathlet from the starting node, terminating when a source, sink or node of degree 3+ is seen"""
        thisPathlet = Pathlet(startingNodeID)
        return thisPathlet
        # TODO - Finish implementing
        # NOTE:
        # A pathlet should terminate when a source, sink, or node of degree 3+ is seen
        # Must be recursive to call the next pathlet construction after a pathlet terminates (via the node stack)
        # Also, account for source starting nodes with multiple outgoing edges and sink nodes with outgoing edges
