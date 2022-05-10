import sys

from numpy import ndarray

from src.AlphaGenetic.Path import Path
from src.Network.FlowNetwork import FlowNetwork


class Individual:
    """Class that defines an individual in the GA population"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, ID: int, network: FlowNetwork, initialAlphaValues: ndarray):
        """Constructor of an Individual instance"""
        # Input Network and ID in the Population
        self.id = ID
        self.network = network
        # Alpha Values (a.k.a. the genotype of the individual)
        self.alphaValues = initialAlphaValues
        # Expressed Network (a.k.a. the phenotype of the individual)
        self.isSolved = False  # Flip true when a relaxed-LP solver runs/returns solution data; flip false when the alpha values array is modified
        self.arcFlows = {}
        self.arcsOpened = {}
        self.srcFlows = []
        self.sinkFlows = []
        self.paths = []  # Data structure for topology-based operators
        # Returned Cost (a.k.a. the fitness of the individual)
        self.trueCost = 0.0
        self.fakeCost = 0.0

    # ============================================
    # ============== HELPER METHODS ==============
    # ============================================
    def setAlphaValues(self, alphaValues: ndarray) -> None:
        """Resets the alpha values to a new array"""
        self.alphaValues = alphaValues
        self.isSolved = False  # Any change to alpha values should reset the isSolved bool

    def resetOutputNetwork(self) -> None:
        """Resets the expressed network (i.e. phenotype) output data in the individual"""
        self.isSolved = False
        self.arcFlows = {}
        self.arcsOpened = {}
        self.srcFlows = []
        self.sinkFlows = []
        self.paths = []
        self.trueCost = 0.0
        self.fakeCost = 0.0

    def resetCostValues(self) -> None:
        """Resets just the cost values (i.e. fitness) of the individual"""
        self.isSolved = False
        self.trueCost = 0.0
        self.fakeCost = 0.0

    # =============================================
    # ============== PATHING METHODS ==============
    # =============================================
    # TODO - Revise pathing methods to handle branching better (i.e. all pathlets, full trees, etc.)
    def computeAllUsedPaths(self) -> None:
        """Computes all the source-sink paths that have a positive flow"""
        if self.isSolved is False:
            print("Cannot compute paths on an unsolved instance!")
        else:
            # For all sources with an assigned flow
            for srcIndex in range(self.network.numSources):
                if self.srcFlows[srcIndex] > 0:
                    # Do a depth first search from the source and use the visited nodes to construct paths
                    src = self.network.sourcesArray[srcIndex]
                    visited = self.doDepthFirstSearch(src)
                    self.constructPaths(visited)

    def doDepthFirstSearch(self, startNode: int, incomingEdge=None) -> tuple:
        """DFS implementation used in computing all used paths"""
        stack = []
        visitedNodes = []
        visitedEdges = []
        # The incomingEdge arg is only used when a complete source/sink path is recursively split due to branching
        if incomingEdge is not None:
            visitedEdges.append(incomingEdge)  # Add edge as (fromNode, toNode) to visited edges
            visitedNodes.append(incomingEdge[0])  # Add fromNode to visited nodes
        # Push starting node onto stack and repeat until stack is empty
        stack.insert(0, startNode)
        while len(stack) > 0:
            node = stack.pop(0)
            # If next node hasn't been visited, visit node
            if node not in visitedNodes:
                visitedNodes.append(node)
                nodeObj = self.network.nodesDict[node]
                # Check all outgoing edges of the node and traverse them to the next node if edge was opened
                for outgoingEdge in nodeObj.outgoingEdges:
                    if self.getEdgeFlow(outgoingEdge) > 0:
                        # Mark the visited edge and add the next node onto the stack
                        visitedEdges.append(outgoingEdge)
                        nextNode = outgoingEdge[1]
                        stack.insert(0, nextNode)
        return visitedNodes, visitedEdges

    def constructPaths(self, visited: tuple) -> None:
        """Constructs all positive flow paths originating from a single source"""
        visitedNodes = visited[0]
        visitedEdges = visited[1]
        # Count number of branch points to determine number of paths
        branchPoints = []
        # Check for branching
        for node in visitedNodes:
            nodeObj = self.network.nodesDict[node]
            if self.numCommonEdges(visitedEdges, nodeObj.outgoingEdges) >= 2:
                branchPoints.append(nodeObj.nodeID)
        # Source has single path
        if len(branchPoints) == 0:
            pathFlowAndCosts = self.calculatePathCostAndFlow(visitedNodes, visitedEdges)
            thisPath = Path(visitedNodes, visitedEdges, pathFlowAndCosts)
            self.paths.append(thisPath)
        # Source has multiple paths/branching (i.e. is a tree) and should be recursively split into arc segments
        else:
            # Branching at the source (i.e. multiple paths leaving from a single source)
            firstBranchID = branchPoints[0]
            firstBranchObj = self.network.nodesDict[firstBranchID]
            if firstBranchObj.nodeType == 0:
                for outgoingEdge in firstBranchObj.outgoingEdges:
                    if self.getEdgeFlow(outgoingEdge) > 0:
                        # For all edges leaving the source with flow, recurse on DFS/Construct Paths
                        recursiveVisited = self.doDepthFirstSearch(outgoingEdge[1], outgoingEdge)
                        self.constructPaths(recursiveVisited)
            # Branching at an intermediate node (i.e. path splits into tree at arbitrary point away from source)
            elif firstBranchObj.nodeType == 2:
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
                pathFlowAndCosts = self.calculatePathCostAndFlow(visitedNodes, visitedEdges)
                pathBefore = Path(visitedNodes, visitedEdges, pathFlowAndCosts)
                self.paths.append(pathBefore)
                # Recurse on DFS/Construct Paths for remaining arc segments
                for outgoingEdge in self.network.nodesDict[branchPoints[0]].outgoingEdges:
                    if self.getEdgeFlow(outgoingEdge) > 0:
                        recursiveVisited = self.doDepthFirstSearch(outgoingEdge[1], outgoingEdge)
                        self.constructPaths(recursiveVisited)

    def getEdgeFlow(self, edge: tuple) -> float:
        """Calculates the combined flow of the edge by considering all parallel arcs"""
        edgeFlow = 0.0
        edgeIndex = self.network.edgesDict[edge]
        for capIndex in range(self.network.numArcCaps):
            edgeFlow += self.arcFlows[(edgeIndex, capIndex)]
        return edgeFlow

    def calculatePathCostAndFlow(self, visitedNodes: list, visitedEdges: list) -> tuple:
        """Computes the total cost of the path and flow on the path"""
        pathFlow = sys.maxsize
        pathFixedCost = 0.0
        pathVariableCost = 0.0
        sinkSrcCost = 0.0
        for edge in visitedEdges:
            thisEdgeFlow = self.getEdgeFlow(edge)
            # Determine the path flow as the bottleneck (i.e. minimum flow) over all edges in the network
            if thisEdgeFlow < pathFlow:
                pathFlow = thisEdgeFlow
            # Update cost on a per arc basis
            edgeIndex = self.network.edgesDict[edge]
            for capIndex in range(self.network.numArcCaps):
                thisArcFlow = self.arcFlows[(edgeIndex, capIndex)]
                if thisArcFlow > 0:
                    cap = self.network.possibleArcCapsArray[capIndex]
                    fixedCost = self.network.arcsDict[(edge[0], edge[1], cap)].fixedCost
                    variableCost = self.network.arcsDict[(edge[0], edge[1], cap)].variableCost
                    pathFixedCost += fixedCost
                    pathVariableCost += variableCost * thisArcFlow
        # Account for source/sink costs
        for node in visitedNodes:
            nodeObj = self.network.nodesDict[node]
            if nodeObj.nodeType == 0:
                for srcIndex in range(self.network.numSources):
                    if self.network.sourcesArray[srcIndex] == node:
                        variableCost = self.network.sourceVariableCostsArray[srcIndex]
                        sinkSrcCost += variableCost * pathFlow
            elif nodeObj.nodeType == 1:
                for sinkIndex in range(self.network.numSinks):
                    if self.network.sinksArray[sinkIndex] == node:
                        variableCost = self.network.sinkVariableCostsArray[sinkIndex]
                        sinkSrcCost += variableCost * pathFlow
        return pathFlow, pathFixedCost, pathVariableCost, sinkSrcCost

    @staticmethod
    def numCommonEdges(visitedEdges: list, nextPossibleEdges: list) -> int:
        """Counts the number of common edges between two list"""
        common = 0
        for edge in visitedEdges:
            if edge in nextPossibleEdges:
                common += 1
        return common

    def printAllPaths(self) -> None:
        """Prints the data for all paths constructed"""
        if len(self.paths) == 0:
            self.computeAllUsedPaths()
        for path in self.paths:
            path.printPathData()
