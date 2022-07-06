import os
import pickle
from typing import List, Dict, Tuple

from src.Graph.CandidateGraph import CandidateGraph


class FlowNetworkSolution:
    """Class that stores the flow network solution to a candidate graph instance"""

    def __init__(self, graph: CandidateGraph, minTargetFlow: float, objectiveValue: float, trueCost: float,
                 sourceFlows: list, sinkFlows: list, arcFlows: dict, solvedBy: str, isOneArcPerEdge: bool,
                 isSourceSinkCapacitated: bool, isSourceSinkCharged: bool, optionalDescription=""):
        """Constructor of a flow network solution instance"""
        # Input attributes
        self.graph: CandidateGraph = graph  # Input candidate graph that the flow network solution solves
        self.minTargetFlow: float = minTargetFlow  # Target flow to capture across the flow network solution
        # Solution Attributes
        self.objectiveValue: float = objectiveValue  # Objective value written by the solver used to generate the solution
        self.trueCost: float = trueCost  # True cost of the solution under the Fixed-Charge Network Flow model
        self.sourceFlows: List[float] = sourceFlows  # List of the flow values assigned to each source, indexed the same as the graph.sourcesArray
        self.sinkFlows: List[float] = sinkFlows  # List of the flow values assigned to each sink, indexed the same as the graph.sinksArray
        self.arcFlows: Dict[Tuple[int, int], float] = arcFlows  # Dictionary mapping (edgeIndex, arcIndex) keys to assigned flow values
        # Metadata/Conditions/Generalizations
        self.name: str = "soln-" + self.graph.name + "_" + str(int(self.minTargetFlow)) + "_" + str(int(self.trueCost)) + solvedBy  # Solution name for saving
        self.solvedBy: str = solvedBy  # Solver method
        self.isOneArcPerEdge: bool = isOneArcPerEdge  # Boolean indicating if the solver considered the constraint that only opens one arc per edge (MILP only)
        self.isSourceSinkCapacitated: bool = isSourceSinkCapacitated  # Boolean indicating if the input graph contained src/sink capacities, which were considered by the solver
        self.isSourceSinkCharged: bool = isSourceSinkCharged  # Boolean indicating if the input graph contained src/sink charges, which were considered by the solver
        self.optionalDescription: str = optionalDescription  # Optional keyword argument to contain additional explanation as needed

    # =================================================
    # ============== DATA IN/OUT METHODS ==============
    # =================================================
    def saveSolution(self) -> None:
        """Saves the solution instance to disc via a pickle dump"""
        # Path management
        currDir = os.getcwd()
        solutionFile = self.name + ".p"
        catPath = os.path.join(currDir, "data/solution_instances", solutionFile)
        print("Saving " + solutionFile + " to: " + catPath)
        # Pickle dump
        pickle.dump(self, open(catPath, "wb"))

    @staticmethod
    def loadSolution(solutionFile: str):
        """Loads a solution instance via a pickle load"""
        # Path management
        currDir = os.getcwd()
        catPath = os.path.join(currDir, "data/solution_instances", solutionFile)
        print("Loading " + solutionFile + " from: " + catPath)
        # Pickle load
        solvedNetwork = pickle.load(open(catPath, "rb"))
        return solvedNetwork
