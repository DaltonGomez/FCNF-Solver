import random

from src.AlphaGeneticSolver.AlphaPath import AlphaPath
from src.AlphaGeneticSolver.AlphaSolver import AlphaSolver
from src.AlphaGeneticSolver.AlphaVisualizer import AlphaVisualizer


class AlphaIndividual:
    """Class that defines an alpha-LP relaxation of a FCFN problem"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, FCFNinstance):
        """Constructor of a AlphaFCNF instance"""
        # Input Attributes
        self.name = FCFNinstance.name + "-Alpha"
        self.FCFN = FCFNinstance  # NOTE: Solution data should not get pushed back to the FCFN instance
        self.alphaValues = []
        self.initializeAlphaValuesRandomly()

        # Solution Data
        self.relaxedSolver = None
        self.isSolved = False
        self.minTargetFlow = 0
        self.fakeCost = 0
        self.totalFlow = 0
        # Opened nodes and edges, where values are tuples of (flow, totalCost)
        self.openedNodesDict = {}
        self.openedEdgesDict = {}
        self.trueCost = 0
        self.paths = []

        # Visualization Data
        self.visualizer = None
        self.visSeed = self.FCFN.visSeed

    # ============================================
    # ============== SOLVER METHODS ==============
    # ============================================
    def executeAlphaSolver(self, minTargetFlow: int) -> None:
        """Solves the FCFN approximately with an alpha-relaxed LP model in CPLEX"""
        self.relaxedSolver = AlphaSolver(self, minTargetFlow)
        self.relaxedSolver.buildModel()
        self.relaxedSolver.solveModel()
        self.relaxedSolver.writeSolution()
        self.calculateTrueCost()
        # self.relaxedSolver.printSolverOverview()

    def calculateTrueCost(self) -> None:
        """Calculates the true cost from the alpha-relaxed LP solution"""
        if self.isSolved is True:
            cost = 0
            for node in self.openedNodesDict.values():
                cost += node[1]
            for edge in self.openedEdgesDict.values():
                cost += edge[1]
            self.trueCost = cost
        else:
            print("The individual must be solved to calculate its true cost!")

    # =============================================
    # ============== PATHING METHODS ==============
    # =============================================
    def allUsedPaths(self) -> None:
        """Computes all the source-sink paths that have a positive flow"""
        if self.isSolved is False:
            print("Cannot compute paths on an unsolved instance!")
        else:
            for node in self.openedNodesDict.keys():
                if node[0] == "s":
                    visited = self.depthFirstSearch(node)
                    self.constructPaths(visited)

    def depthFirstSearch(self, startNode: str, incomingEdge="") -> list:
        """DFS implementation used in computing all used paths"""
        stack = []
        visitedNodes = []
        visitedEdges = []
        # The incomingEdge arg is only used when a complete source/sink path is recursively split due to branching
        if incomingEdge != "":
            visitedEdges.append(incomingEdge)
            visitedNodes.append(self.FCFN.edgesDict[incomingEdge].fromNode)
        # Push starting node onto stack and repeat until stack is empty
        stack.insert(0, startNode)
        while len(stack) > 0:
            node = stack.pop(0)
            # If next node hasn't been visited, visit node
            if node not in visitedNodes:
                visitedNodes.append(node)
                nodeObj = self.FCFN.nodesDict[node]
                # Check all outgoing edges of the node and traverse them to the next node if edge was opened
                for outgoingEdge in nodeObj.outgoingEdges:
                    if outgoingEdge in self.openedEdgesDict.keys():
                        edgeObj = self.FCFN.edgesDict[outgoingEdge]
                        # Mark the visited edge and add the next node onto the stack
                        visitedEdges.append(outgoingEdge)
                        nextNode = edgeObj.toNode
                        stack.insert(0, nextNode)
        return [visitedNodes, visitedEdges]

    def constructPaths(self, visited: list) -> None:
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
            thisPath = AlphaPath(visitedNodes, visitedEdges, self)
            self.paths.append(thisPath)
        # Source has multiple paths/branching (i.e. is a tree) and should be recursively split into arc segments
        else:
            # Branching at the source (i.e. multiple paths leaving from a single source)
            if branchPoints[0][0] == "s":
                for outgoingEdge in self.FCFN.nodesDict[branchPoints[0]].outgoingEdges:
                    if outgoingEdge in self.openedEdgesDict.keys():
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
                pathBefore = AlphaPath(nodesBeforeBranch, edgesBeforeBranch, self)
                self.paths.append(pathBefore)
                # Recurse on DFS/Construct Paths for remaining arc segments
                for outgoingEdge in self.FCFN.nodesDict[branchPoints[0]].outgoingEdges:
                    if outgoingEdge in self.openedEdgesDict.keys():
                        recursiveVisited = self.depthFirstSearch(self.FCFN.edgesDict[outgoingEdge].toNode,
                                                                 outgoingEdge)
                        self.constructPaths(recursiveVisited)

    @staticmethod
    def numCommonEdges(visitedEdges: list, nextPossibleEdges: list) -> int:
        """Counts the number of common edges between two list"""
        common = 0
        for edge in visitedEdges:
            if edge in nextPossibleEdges:
                common += 1
        return common

    # ==========================================================
    # ============== ALPHA INITIALIZATION METHODS ==============
    # ==========================================================
    def initializeAlphaValuesConstantly(self, constant: float) -> None:
        """Initializes all alpha values to the input constant"""
        random.seed()
        theseAlphaValues = []
        for i in range(self.FCFN.numEdges):
            theseAlphaValues.append(constant)
        self.alphaValues = theseAlphaValues

    def initializeAlphaValuesRandomly(self, lowerBound=0.0, upperBound=1.0) -> None:
        """Randomly initializes alpha values on [LB, UB] range, which defaults to [0, 1]"""
        random.seed()
        theseAlphaValues = []
        for i in range(self.FCFN.numEdges):
            theseAlphaValues.append(random.uniform(lowerBound, upperBound))
        self.alphaValues = theseAlphaValues

    # =========================================================
    # ============== VISUALIZATION/PRINT METHODS ==============
    # =========================================================
    def visualizeAlphaNetwork(self, frontCatName="", endCatName="", graphType="fullGraph") -> None:
        """Draws the Fixed Charge Flow Network instance using the PyVis package and a NetworkX conversion"""
        if self.visualizer is None:
            self.visualizer = AlphaVisualizer(self, graphType)
            self.visualizer.drawGraph(frontCatName + self.name + endCatName)
        else:
            self.visualizer.drawGraph(frontCatName + self.name + endCatName)

    def calculateTotalCostByPaths(self) -> int:
        """Computes the true cost of the complete network by the paths and not the nodes/edges"""
        costByPaths = 0
        for path in self.paths:
            costByPaths += path.totalCost
        return costByPaths

    def pathsVsElementsCost(self) -> None:
        """Compares true costs found by pathing vs. enumerating individual elements"""
        edgeCostByPaths = 0
        sourceCostByPaths = 0
        sinkCostByPaths = 0
        for path in self.paths:
            edgeCostByPaths += path.routingCost
            sourceCostByPaths += path.startCost
            sinkCostByPaths += path.endCost
        trueEdgeCost = 0
        for edgeValue in self.openedEdgesDict.values():
            trueEdgeCost += edgeValue[1]
        trueSourceCost = 0
        trueSinkCost = 0
        for node in self.openedEdgesDict:
            if node[0][0] == "s":
                trueSourceCost += node[1][1]
            if node[0][0] == "t":
                trueSinkCost += node[1][1]
        print("TRUE TOTAL COST = " + str(self.trueCost))
        print("PATH TOTAL COST = " + str(self.calculateTotalCostByPaths()))
        print("True Edge Cost = " + str(trueEdgeCost))
        print("Path Edge Cost = " + str(edgeCostByPaths))
        print("True Source Cost = " + str(trueSourceCost))
        print("Path Source Cost = " + str(sourceCostByPaths))
        print("True Sink Cost = " + str(trueSinkCost))
        print("Path Sink Cost = " + str(sinkCostByPaths))

    def printAllPathData(self) -> None:
        """Prints all the data for each path in the network"""
        if len(self.paths) == 0:
            self.allUsedPaths()
        for path in self.paths:
            path.printPathData()
