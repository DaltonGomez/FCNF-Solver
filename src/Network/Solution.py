import os
import pickle

from src.Network.FlowNetwork import FlowNetwork


class Solution:
    """Class that stores the solution to an FCFN instance"""

    def __init__(self, network: FlowNetwork, minTargetFlow: float, objectiveValue: float, trueCost: float,
                 sourceFlows: list, sinkFlows: list, arcFlows: dict, arcsOpened: dict, solvedBy: str,
                 isOneArcPerEdge: bool, isSrcSinkConstrained: bool, isSrcSinkCharged: bool, optionalDescription=""):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.network = network
        self.minTargetFlow = minTargetFlow
        # Solution Attributes
        self.objectiveValue = objectiveValue
        self.trueCost = trueCost
        self.sourceFlows = sourceFlows
        self.sinkFlows = sinkFlows
        self.arcFlows = arcFlows
        self.arcsOpened = arcsOpened
        # Metadata/Conditions/Generalizations
        self.name = "soln-" + self.network.name + "_" + str(int(self.minTargetFlow)) + "_" + str(
            int(self.trueCost)) + solvedBy
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
        parentDir = os.path.abspath(os.path.join(currDir, os.pardir))
        solutionFile = self.name + ".p"
        catPath = os.path.join(parentDir, "data/solution_instances", solutionFile)
        print("Saving " + solutionFile + " to: " + catPath)
        # Pickle dump
        pickle.dump(self, open(catPath, "wb"))

    @staticmethod
    def loadSolution(solutionFile: str):
        """Loads a solution instance via a pickle load"""
        # Path management
        currDir = os.getcwd()
        parentDir = os.path.abspath(os.path.join(currDir, os.pardir))
        catPath = os.path.join(parentDir, "data/solution_instances", solutionFile)
        print("Loading " + solutionFile + " from: " + catPath)
        # Pickle load
        solvedNetwork = pickle.load(open(catPath, "rb"))
        return solvedNetwork
