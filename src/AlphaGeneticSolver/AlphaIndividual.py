import copy
import random

from src.AlphaGeneticSolver.AlphaSolver import AlphaSolver
from src.AlphaGeneticSolver.AlphaVisualizer import AlphaVisualizer


class AlphaIndividual:
    """Class that defines an alpha-LP reduction of a FCFN problem"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, individualNum: int, FCFNinstance):
        """Constructor of a AlphaFCNF instance"""
        # Input Attributes
        self.name = FCFNinstance.name + "-Alpha" + str(individualNum)
        self.idNumber = individualNum
        self.FCNF = copy.deepcopy(FCFNinstance)
        self.alphaValues = None
        self.initializeAlphaValuesConstantly(0.5)

        # Solution Data
        self.relaxedSolver = None
        self.isSolved = False
        self.minTargetFlow = 0
        self.fakeCost = 0
        self.totalFlow = 0
        self.trueCost = 0

        # Visualization Data
        self.visualizer = None
        self.visSeed = 1

    # ============================================
    # ============== SOLVER METHODS ==============
    # ============================================
    def executeAlphaSolver(self, minTargetFlow: int):
        """Solves the FCFN approximately with an alpha-reduced LP model in CPLEX"""
        if self.relaxedSolver is None:
            self.relaxedSolver = AlphaSolver(self,
                                             minTargetFlow)  # FYI- ExactSolver constructor does not have FCFN type hint
            self.relaxedSolver.buildModel()
            self.relaxedSolver.solveModel()
            self.relaxedSolver.writeSolution()
            self.calculateTrueCost()
            self.relaxedSolver.printSolverOverview()
        elif self.relaxedSolver.isRun is True and self.isSolved is False:
            print("No feasible solution exists for the network and target!")
        elif self.relaxedSolver.isRun is True and self.isSolved is True:
            print("Model is already solved- Call print solution to view solution!")

    def calculateTrueCost(self):
        """Calculates the true cost from the alpha-relaxed LP solution"""
        if self.isSolved is True:
            cost = 0
            for node in self.FCNF.nodesDict:
                nodeObj = self.FCNF.nodesDict[node]
                cost += nodeObj.totalCost
            for edge in self.FCNF.edgesDict:
                edgeObj = self.FCNF.edgesDict[edge]
                if edgeObj.flow > 0:
                    trueEdgeCost = edgeObj.flow * edgeObj.variableCost + edgeObj.fixedCost
                    cost += trueEdgeCost
            self.trueCost = cost
        else:
            print("The individual must be solved to calculate its true cost!")

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def visualizeAlphaNetwork(self, catName=""):
        """Draws the Fixed Charge Flow Network instance using the PyVis package and a NetworkX conversion"""
        if self.visualizer is None:
            self.visualizer = AlphaVisualizer(self)
            self.visualizer.drawGraph(self.name + catName)

    def initializeAlphaValues(self, initializationMethod: str, constant=0):
        """Initializes an individual's alpha values using the input method"""
        if initializationMethod == "random":
            self.initializeAlphaValuesRandomly()
        elif initializationMethod == "constant":
            self.initializeAlphaValuesConstantly(constant)

    def initializeAlphaValuesConstantly(self, constant: float):
        """Initializes all alpha values to the input constant"""
        random.seed()
        theseAlphaValues = []
        for i in range(self.FCNF.numEdges):
            theseAlphaValues.append(constant)
        self.alphaValues = theseAlphaValues

    def initializeAlphaValuesRandomly(self):
        """Randomly initializes alpha values on [0, 1]"""
        random.seed()
        theseAlphaValues = []
        for i in range(self.FCNF.numEdges):
            theseAlphaValues.append(random.random())
        self.alphaValues = theseAlphaValues
