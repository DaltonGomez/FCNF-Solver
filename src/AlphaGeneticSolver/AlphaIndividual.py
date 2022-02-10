import copy
import random

from src.AlphaGeneticSolver.AlphaSolver import AlphaSolver
from src.AlphaGeneticSolver.AlphaVisualizer import AlphaVisualizer


class AlphaIndividual:
    """Class that defines an alpha-LP reduction of a FCFN problem"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, FCFNinstance):
        """Constructor of a AlphaFCNF instance"""
        # Input Attributes
        self.name = FCFNinstance.name + "-Alpha"
        self.FCFN = copy.deepcopy(FCFNinstance)
        self.alphaValues = None
        self.initializeAlphaValuesRandomly()

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
        self.relaxedSolver = AlphaSolver(self, minTargetFlow)
        self.relaxedSolver.buildModel()
        self.relaxedSolver.solveModel()
        self.relaxedSolver.writeSolution()
        self.calculateTrueCost()
        # self.relaxedSolver.printSolverOverview()

    def calculateTrueCost(self):
        """Calculates the true cost from the alpha-relaxed LP solution"""
        if self.isSolved is True:
            cost = 0
            for node in self.FCFN.nodesDict:
                nodeObj = self.FCFN.nodesDict[node]
                cost += nodeObj.totalCost
            for edge in self.FCFN.edgesDict:
                edgeObj = self.FCFN.edgesDict[edge]
                if edgeObj.flow > 0:
                    trueEdgeCost = edgeObj.flow * edgeObj.variableCost + edgeObj.fixedCost
                    cost += trueEdgeCost
            self.trueCost = cost
        else:
            print("The individual must be solved to calculate its true cost!")

    # ==========================================================
    # ============== ALPHA INITIALIZATION METHODS ==============
    # ==========================================================
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
        for i in range(self.FCFN.numEdges):
            theseAlphaValues.append(constant)
        self.alphaValues = theseAlphaValues

    def initializeAlphaValuesRandomly(self):
        """Randomly initializes alpha values on [0, 1]"""
        random.seed()
        theseAlphaValues = []
        for i in range(self.FCFN.numEdges):
            theseAlphaValues.append(random.random())
        self.alphaValues = theseAlphaValues

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def visualizeAlphaNetwork(self, frontCatName="", endCatName=""):
        """Draws the Fixed Charge Flow Network instance using the PyVis package and a NetworkX conversion"""
        if self.visualizer is None:
            self.visualizer = AlphaVisualizer(self)
            self.visualizer.drawGraph(frontCatName + self.name + endCatName)
        else:
            self.visualizer.drawGraph(frontCatName + self.name + endCatName)
