import copy
import random

from src.AlphaGeneticSolver.AlphaPath import AlphaPath
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

    def depthFirstSearch(self, startNode: str, incomingEdge=""):
        """DFS implementation used in computing all used paths"""
        stack = []
        visitedNodes = []
        visitedEdges = []
        # The incomingEdge arg is only used when a complete source/sink path is recursively split due to branching
        if incomingEdge != "":
            visitedEdges.append(incomingEdge)
            visitedNodes.append(self.FCFN.edgesDict[incomingEdge].fromNode)
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
        return [visitedNodes, visitedEdges]

    def constructPaths(self, visited: list):
        """Constructs all positive flow paths originating from a single source"""
        visitedNodes = visited[0]
        visitedEdges = visited[1]
        # Count number of branch points to determine number of paths
        branchPoints = []
        # Check for branching
        for node in visitedNodes:
            nodeObj = self.FCFN.nodesDict[node]
            if self.numCommonEdges(visitedEdges, nodeObj.outgoingEdges) >= 2:
                branchPoints.append(nodeObj.nodeID)
        # Source has single path
        if len(branchPoints) == 0:
            thisPath = AlphaPath(visitedNodes, visitedEdges, self.FCFN)
            self.paths.append(thisPath)
        # Source has multiple paths/branching (i.e. is a tree) and should be recursively split into arc segments
        else:
            # Branching at the source (i.e. multiple paths leaving from a single source)
            if branchPoints[0][0] == "s":
                for outgoingEdge in self.FCFN.nodesDict[branchPoints[0]].outgoingEdges:
                    if self.FCFN.edgesDict[outgoingEdge].flow > 0:
                        # For all edges leaving the source with flow, recurse on DFS/Construct Paths
                        recursiveVisited = self.depthFirstSearch(self.FCFN.edgesDict[outgoingEdge].toNode,
                                                                 outgoingEdge)
                        self.constructPaths(recursiveVisited)
            # Branching at an intermediate node (i.e. path splits into tree at arbitrary point away from source)
            elif branchPoints[0][0] == "n":
                # Create path for everything before intermediate branch
                currentNode = visitedNodes[0]
                nodesBeforeBranch = []
                edgesBeforeBranch = []
                # Pop nodes and edges up to branch
                while currentNode != branchPoints[0]:
                    nodesBeforeBranch.append(visitedNodes.pop(0))
                    edgesBeforeBranch.append(visitedEdges.pop(0))
                    currentNode = visitedNodes[0]
                # Package as an arc segment
                pathBefore = AlphaPath(nodesBeforeBranch, edgesBeforeBranch, self.FCFN)
                self.paths.append(pathBefore)
                # Recurse on DFS/Construct Paths for remaining arc segments
                for outgoingEdge in self.FCFN.nodesDict[branchPoints[0]].outgoingEdges:
                    if self.FCFN.edgesDict[outgoingEdge].flow > 0:
                        recursiveVisited = self.depthFirstSearch(self.FCFN.edgesDict[outgoingEdge].toNode,
                                                                 outgoingEdge)
                        self.constructPaths(recursiveVisited)

    @staticmethod
    def numCommonEdges(visitedEdges: list, nextPossibleEdges: list):
        """Counts the number of common edges between two list"""
        common = 0
        for edge in visitedEdges:
            if edge in nextPossibleEdges:
                common += 1
        return common

    def calculateTotalCostByPaths(self):
        """Computes the true cost of the complete network by the paths and not the nodes/edges"""
        costByPaths = 0
        for path in self.paths:
            costByPaths += path.totalCost
        return costByPaths

    def pathsVsElementsCost(self):
        """Compares true costs found by pathing vs. enumerating individual elements"""
        edgeCostByPaths = 0
        sourceCostByPaths = 0
        sinkCostByPaths = 0
        for path in self.paths:
            edgeCostByPaths += path.routingCost
            sourceCostByPaths += path.startCost
            sinkCostByPaths += path.endCost
        trueEdgeCost = 0
        for i in range(self.FCFN.numEdges):
            trueEdgeCost += self.FCFN.edgesDict["e" + str(i)].totalCost
        trueSourceCost = 0
        for s in range(self.FCFN.numSources):
            trueSourceCost += self.FCFN.nodesDict["s" + str(s)].totalCost
        trueSinkCost = 0
        for t in range(self.FCFN.numSinks):
            trueSinkCost += self.FCFN.nodesDict["t" + str(t)].totalCost
        print("TRUE TOTAL COST = " + str(self.trueCost))
        print("PATH TOTAL COST = " + str(self.calculateTotalCostByPaths()))
        print("True Edge Cost = " + str(trueEdgeCost))
        print("Path Edge Cost = " + str(edgeCostByPaths))
        print("True Source Cost = " + str(trueSourceCost))
        print("Path Source Cost = " + str(sourceCostByPaths))
        print("True Sink Cost = " + str(trueSinkCost))
        print("Path Sink Cost = " + str(sinkCostByPaths))

    def printAllPathData(self):
        """Prints all the data for each path in the network"""
        for path in self.paths:
            path.printPathData()

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
