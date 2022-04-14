import os
import pickle

from src.Network.FlowNetwork import FlowNetwork


class Solution:
    """Class that stores the solution to an FCFN instance"""

    def __init__(self, network: FlowNetwork, minTargetFlow: int, objectiveValue: float, sourceFlows: list,
                 sinkFlows: list, arcFlows: dict, arcsOpened: dict, solvedBy: str, isOneArcPerEdge: bool,
                 isSrcSinkConstrained: bool, isSrcSinkCharged: bool, optionalDescription=""):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.network = network
        self.minTargetFlow = minTargetFlow
        # Solution Attributes
        self.objectiveValue = objectiveValue
        self.sourceFlows = sourceFlows
        self.sinkFlows = sinkFlows
        self.arcFlows = arcFlows
        self.arcsOpened = arcsOpened
        # Metadata/Conditions/Generalizations
        self.solvedBy = solvedBy
        self.isOneArcPerEdge = isOneArcPerEdge
        self.isSrcSinkConstrained = isSrcSinkConstrained
        self.isSrcSinkCharged = isSrcSinkCharged
        self.optionalDescription = optionalDescription

    # =================================================
    # ============== DATA IN/OUT METHODS ==============
    # =================================================
    def saveSolution(self) -> None:
        """Saves the solution instance to disc via a pickle dump"""
        # Path management
        currDir = os.getcwd()
        solutionFile = "Soln_" + self.network.name + "_" + self.solvedBy + ".p"
        catPath = os.path.join(currDir, "solutionInstances", solutionFile)
        print("Saving " + solutionFile + " to: " + catPath)
        # Pickle dump
        pickle.dump(self, open(catPath, "wb"))

    @staticmethod
    def loadSolution(solutionFile: str):
        """Loads a solution instance via a pickle load"""
        # Path management
        currDir = os.getcwd()
        catPath = os.path.join(currDir, "solutionInstances", solutionFile)
        print("Loading " + solutionFile + " from: " + catPath)
        # Pickle load
        solvedNetwork = pickle.load(open(catPath, "rb"))
        return solvedNetwork
