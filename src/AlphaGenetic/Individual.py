from numpy import ndarray

from AlphaGenetic.Path import Path
from src.Network.FlowNetwork import FlowNetwork


class Individual:
    """Class that defines an individual in the GA population"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, network: FlowNetwork, initialAlphaValues: ndarray):
        """Constructor of an Individual instance"""
        # Input Network
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
    # TODO - THESE METHOD ARE FROM VERSION_1 AND NEED UPDATING
    def allUsedPaths(self) -> None:
        """Computes all the source-sink paths that have a positive flow"""
        if self.isSolved is False:
            print("Cannot compute paths on an unsolved instance!")
        else:
            for src in self.srcFlows:
                if src > 0:
                    srcID = self.network.sourcesArray[src]
                    visited = self.depthFirstSearch(srcID)
                    self.constructPaths(visited)

    def depthFirstSearch(self, startNode: str, incomingEdge="") -> list:
        """DFS implementation used in computing all used paths"""
        stack = []
        visitedNodes = []
        visitedEdges = []
        # The incomingEdge arg is only used when a complete source/sink path is recursively split due to branching
        if incomingEdge != "":
            visitedEdges.append(incomingEdge)
            visitedNodes.append(self.network.edgesDict[incomingEdge].fromNode)
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
                    if outgoingEdge in self.arcsOpened.keys():
                        edgeObj = self.network.edgesDict[outgoingEdge]
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
            nodeObj = self.network.nodesDict[node]
            if self.numCommonEdges(visitedEdges, nodeObj.outgoingEdges) >= 2:
                branchPoints.append(nodeObj.nodeID)
        # Source has single path
        if len(branchPoints) == 0:
            thisPath = Path(self.network, visitedNodes, visitedEdges)
            self.paths.append(thisPath)
        # Source has multiple paths/branching (i.e. is a tree) and should be recursively split into arc segments
        else:
            # Branching at the source (i.e. multiple paths leaving from a single source)
            if branchPoints[0][0] == "s":
                for outgoingEdge in self.network.nodesDict[branchPoints[0]].outgoingEdges:
                    if outgoingEdge in self.arcsOpened.keys():
                        # For all edges leaving the source with flow, recurse on DFS/Construct Paths
                        recursiveVisited = self.depthFirstSearch(self.network.edgesDict[outgoingEdge].toNode,
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
                pathBefore = Path(self.network, nodesBeforeBranch, edgesBeforeBranch)
                self.paths.append(pathBefore)
                # Recurse on DFS/Construct Paths for remaining arc segments
                for outgoingEdge in self.network.nodesDict[branchPoints[0]].outgoingEdges:
                    if outgoingEdge in self.arcsOpened.keys():
                        recursiveVisited = self.depthFirstSearch(self.network.edgesDict[outgoingEdge].toNode,
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
