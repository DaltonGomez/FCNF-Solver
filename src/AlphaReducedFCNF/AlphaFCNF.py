import copy
import random

from src.FixedCostNetworkFlowSolver.FCNF import FCNF


class AlphaFCNF:
    """Class that defines an alpha-LP reduction of a FCNF problem"""

    # TODO - Continue debugging how the optimal alpha values

    def __init__(self, FCNFinstance: FCNF):
        """Constructor of a AlphaFCNF instance"""
        # Input Attributes
        self.name = FCNFinstance.name + "-Alpha"
        self.FCNF = copy.deepcopy(FCNFinstance)
        self.alphaValues = self.computeOptimalAlphaValues()

        # Solution Data
        self.solved = False
        self.minTargetFlow = 0
        # TODO- Modify so that we are calculating the true cost, not the "fake cost" returned by the LP
        self.totalCost = 0
        self.totalFlow = 0

        # Deterministic seed for consistent visualization
        self.visSeed = FCNFinstance.visSeed

    def initializeAlphaValuesRandomly(self):
        """Randomly initializes alpha values on [0, 1]"""
        random.seed()
        alphaValues = []
        for i in range(self.FCNF.numEdges):
            alphaValues.append(random.random())
        return alphaValues

    def computeOptimalAlphaValues(self):
        """Computes the optimal alpha values from a solved MILP of a FCNF instance"""
        alphaValues = []
        if self.FCNF.solved is True:
            for edge in self.FCNF.edgesDict:
                edgeObj = self.FCNF.edgesDict[edge]
                if edgeObj.flow == 0:
                    # TODO- Consider more the difference between these two lines
                    alphaValues.append(1)
                    # alphaValues.append(0)
                else:
                    alphaValues.append(1 / edgeObj.flow)
        else:
            print("Must solve the original FCNF optimally before computing optimal alpha values")
        return alphaValues

    def resetOriginalFCNFSolution(self):
        """Resets the saved solution data in the original FCNF solution"""
        self.FCNF.solved = False
        self.FCNF.minTargetFlow = 0
        self.FCNF.totalCost = 0
        self.FCNF.totalFlow = 0
        for node in self.FCNF.nodesDict:
            nodeObj = self.FCNF.nodesDict[node]
            nodeObj.opened = False
            nodeObj.flow = 0
            nodeObj.totalCost = 0
        for edge in self.FCNF.edgesDict:
            edgeObj = self.FCNF.edgesDict[edge]
            edgeObj.opened = False
            edgeObj.fixedCost = 0
            edgeObj.variableCost = 0
            edgeObj.capacity = 0
            edgeObj.flow = 0
            edgeObj.totalCost = 0
