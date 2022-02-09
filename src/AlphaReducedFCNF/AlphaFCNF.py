import copy
import random

from src.FixedCostNetworkFlowSolver.FCNF import FCNF


class AlphaFCNF:
    """Class that defines an alpha-LP reduction of a FCNF problem"""

    def __init__(self, FCNFinstance: FCNF):
        """Constructor of a AlphaFCNF instance"""
        # Input Attributes
        self.name = FCNFinstance.name + "-Alpha"
        self.FCNF = copy.deepcopy(FCNFinstance)
        self.alphaValues = self.initializeAlphaValuesRandomly()

        # Solution Data
        self.solved = False
        self.minTargetFlow = 0
        self.totalCost = 0
        self.totalFlow = 0
        self.trueCost = 0

        # Deterministic seed for consistent visualization
        self.visSeed = FCNFinstance.visSeed

    def initializeAlphaValuesRandomly(self):
        """Randomly initializes alpha values on [0, 1]"""
        random.seed()
        alphaValues = []
        for i in range(self.FCNF.numEdges):
            alphaValues.append(random.random())
        return alphaValues

    def calculateTrueCost(self):
        """Calculates the true cost from the alpha-relaxed LP solution"""
        for node in self.FCNF.nodesDict:
            nodeObj = self.FCNF.nodesDict[node]
            self.trueCost += nodeObj.totalCost
        for edge in self.FCNF.edgesDict:
            edgeObj = self.FCNF.edgesDict[edge]
            if edgeObj.flow > 0:
                trueEdgeCost = edgeObj.flow * int(edgeObj.variableCost) + int(edgeObj.fixedCost)
                self.trueCost += trueEdgeCost
        self.totalCost = self.trueCost
