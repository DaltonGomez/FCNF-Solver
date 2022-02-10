import copy
import random

from src.AlphaGeneticSolver.AlphaPath import AlphaPath
from src.AlphaGeneticSolver.AlphaSolver import AlphaSolver
from src.AlphaGeneticSolver.AlphaVisualizer import AlphaVisualizer


# noinspection PyMethodMayBeStatic
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
        self.paths = []

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

    # =============================================
    # ============== PATHING METHODS ==============
    # =============================================
    def allUsedPaths(self):
        """Computes all the source-sink paths that have a positive flow"""
        for i in range(self.FCFN.numSources):
            source = "s" + str(i)
            srcObj = self.FCFN.nodesDict[source]
            if srcObj.flow > 0:
                visited = self.depthFirstSearch(source)
                self.constructPaths(visited)

    def depthFirstSearch(self, startNode: str):
        """DFS implementation used in computing all used paths"""
        stack = []
        visitedNodes = []
        visitedEdges = []
        stack.insert(0, startNode)
        while len(stack) > 0:
            node = stack.pop(0)
            if node not in visitedNodes:
                visitedNodes.append(node)
                nodeObj = self.FCFN.nodesDict[node]
                for outgoingEdge in nodeObj.outgoingEdges:
                    edgeObj = self.FCFN.edgesDict[outgoingEdge]
                    if edgeObj.flow > 0:
                        visitedEdges.append(outgoingEdge)
                        nextNode = edgeObj.toNode
                        stack.insert(0, nextNode)
        print("\n")
        print(visitedNodes)
        print(visitedEdges)
        return [visitedNodes, visitedEdges]

    def constructPaths(self, visited: list):
        """Constructs all positive flow paths originating from a single source"""
        visitedNodes = visited[0]
        visitedEdges = visited[1]
        # Count number of branch points to determine number of paths
        branchPoints = []
        # Check for branching at the source
        sourceObj = self.FCFN.nodesDict[visitedNodes[0]]
        if self.numCommonEdges(visitedEdges, sourceObj.outgoingEdges) >= 2:
            branchPoints.append(sourceObj.nodeID)
        # Check for branching at an intermediate node
        for edge in visitedEdges:
            toNode = self.FCFN.edgesDict[edge].toNode
            nextPossibleEdges = self.FCFN.nodesDict[toNode].outgoingEdges
            if self.numCommonEdges(visitedEdges, nextPossibleEdges) >= 2:
                branchPoints.append(edge)
        print(branchPoints)
        # Source has single path
        if len(branchPoints) == 0:
            thisPath = AlphaPath(visitedNodes, visitedEdges, True, self.FCFN)
            self.paths.append(thisPath)
            thisPath.printPathData()
        # Source has multiple paths/is a tree
        else:
            thisPath = AlphaPath(visitedNodes, visitedEdges, False, self.FCFN)
            self.paths.append(thisPath)
            thisPath.printPathData()
            # TODO - Update to recursively decompose trees into their component sub-paths by calling the DFS method

    def numCommonEdges(self, visitedEdges: list, nextPossibleEdges: list):
        """Counts the number of common edges between two list"""
        common = 0
        for edge in visitedEdges:
            if edge in nextPossibleEdges:
                common += 1
        return common

    # ==========================================================
    # ============== ALPHA INITIALIZATION METHODS ==============
    # ==========================================================
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

    def initializeAlphaValuesRandomlyOnRange(self, lowerBound: float, upperBound: float):
        """Randomly initializes alpha values on [LB, UB] range"""
        random.seed()
        theseAlphaValues = []
        for i in range(self.FCFN.numEdges):
            theseAlphaValues.append(random.uniform(lowerBound, upperBound))
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
