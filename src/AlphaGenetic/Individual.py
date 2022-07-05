
import sys
from typing import Dict, Tuple, List

from numpy import ndarray

from FlowNetwork.FlowNetworkSolution import FlowNetworkSolution
from src.AlphaGenetic.Path import Path
from src.FlowNetwork.CandidateGraph import CandidateGraph


class Individual:
    """Class that defines an individual in the GA population"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, ID: int, graph: CandidateGraph, initialAlphaValues: ndarray):
        """Constructor of an Individual instance"""
        # Input candidate graph and ID in the population
        self.id: int = ID  # Integer ID specifying the individuals index in the population
        self.graph: CandidateGraph = graph  # Input candidate graph instance
        # Alpha Values (a.k.a. the genotype of the individual)
        self.alphaValues: ndarray = initialAlphaValues  # Numpy array of alpha values assigned to the individual at initialization
        # Expressed FlowNetwork (a.k.a. the phenotype of the individual)
        self.isSolved: bool = False  # Flip true when a relaxed-LP solver runs/returns solution data; flip false when the alpha values array is modified
        self.arcFlows: Dict[Tuple[int, int], float] = {}  # Dictionary mapping (edgeIndex, capIndex) keys to values of assigned flow
        self.srcFlows: List[float] = []  # List of assigned flows on each source, where indices match graph.sourcesArray
        self.sinkFlows: List[float] = []  # List of assigned flows on each sink, where indices match graph.sourcesArray
        # Returned Cost (a.k.a. the fitness of the individual)
        self.trueCost: float = 0.0  # True cost computed by using the assigned flows and the Fixed Charge Network Flow model formulation
        self.fakeCost: float = 0.0  # Actual objective value returned by the solver (used only for debugging purposes)

        # TODO - Decide if pathing is to remain or be removed
        self.paths = []

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
        self.srcFlows = []
        self.sinkFlows = []
        self.arcFlows = {}
        self.trueCost = 0.0
        self.fakeCost = 0.0
        self.paths = []

    def resetCostValues(self) -> None:
        """Resets just the cost values (i.e. fitness) of the individual"""
        self.isSolved = False
        self.trueCost = 0.0
        self.fakeCost = 0.0

    def writeIndividualAsSolution(self, minTargetFlow: float) -> FlowNetworkSolution:
        """Writes the current expressed network as a solution instance"""
        if self.isSolved is False:
            print("You must solve the individual before writing a solution!")
        else:
            print("Writing solution from individual #" + str(self.id) + "...")
            thisSolution = FlowNetworkSolution(self.graph, minTargetFlow, self.fakeCost, self.trueCost,
                                               self.srcFlows, self.sinkFlows, self.arcFlows, "Alpha_GA", False,
                                               self.graph.isSourceSinkCapacitated, self.graph.isSourceSinkCharged)
            return thisSolution

    # TODO - Improve pathing methods- Currently not working!
    # =============================================
    # ============== PATHING METHODS ==============
    # =============================================
    # TODO - Improve pathing methods- Currently not working!
    def computeAllUsedPaths(self) -> None:
        """Computes all the pathlets that have a positive flow"""
        if self.isSolved is False:
            print("Cannot compute paths on an unsolved instance!")
        else:
            edgesAssignedToPaths = set()  # Attempts to prevent looping in the buildPathlet() recursion
            # For all sources with an assigned flow
            for srcIndex in range(self.graph.numSources):
                if self.srcFlows[srcIndex] > 0:
                    src = self.graph.sourcesArray[srcIndex]
                    srcObj = self.graph.nodesDict[src]
                    # For all outgoing edges with flow, build pathlets
                    for outgoingEdge in srcObj.outgoingEdges:
                        if self.getEdgeFlow(outgoingEdge) > 0:
                            self.buildPathlet(src, outgoingEdge, edgesAssignedToPaths)

    def buildPathlet(self, startingNode: int, startingEdge: tuple, edgesAssignedToPaths: set) -> None:
        """Builds a pathlet from a starting node in a particular direction"""
        visitedNodes = [startingNode]
        visitedEdges = [startingEdge]
        toNode = startingEdge[1]
        toNodeObj = self.graph.nodesDict[toNode]
        # While the pathlet hasn't been interrupted, travel it
        while self.isEndOfPathlet(toNode) is not True:
            for edge in toNodeObj.outgoingEdges:
                if self.getEdgeFlow(edge) > 0:
                    visitedNodes.append(toNode)
                    visitedEdges.append(edge)
                    toNode = edge[1]
                    toNodeObj = self.graph.nodesDict[toNode]
        # When the pathlet is interrupted, construct a Path object
        visitedNodes.append(toNode)
        if self.isDuplicatePathlet(visitedNodes) is not True:
            pathFlowAndCosts = self.calculatePathCostAndFlow(visitedNodes, visitedEdges)
            thisPathlet = Path(visitedNodes, visitedEdges, pathFlowAndCosts)
            self.paths.append(thisPathlet)
            for edge in visitedEdges:
                edgesAssignedToPaths.add(edge)
        # Recurse on buildPathlet() if the termination of the last pathlet should initiate a new pathlet
        # If the toNode is a branch point
        outgoingFlows = 0
        for edge in toNodeObj.outgoingEdges:
            if self.getEdgeFlow(edge) > 0:
                outgoingFlows += 1
        if outgoingFlows > 1:
            for edge in toNodeObj.outgoingEdges:
                if self.getEdgeFlow(edge) > 0 and edge not in edgesAssignedToPaths:
                    self.buildPathlet(toNode, edge, edgesAssignedToPaths)
        # If the toNode is a confluence point
        incomingFlows = 0
        for edge in toNodeObj.incomingEdges:
            if self.getEdgeFlow(edge) > 0:
                incomingFlows += 1
        if incomingFlows > 1:
            for edge in toNodeObj.outgoingEdges:
                if self.getEdgeFlow(edge) > 0 and edge not in edgesAssignedToPaths:
                    self.buildPathlet(toNode, edge, edgesAssignedToPaths)
        # If the toNode is a sink that has positive outgoing flow
        if toNodeObj.nodeType == 1:
            for edge in toNodeObj.outgoingEdges:
                if self.getEdgeFlow(edge) > 0 and edge not in edgesAssignedToPaths:
                    self.buildPathlet(toNode, edge, edgesAssignedToPaths)

    def isEndOfPathlet(self, toNode: int) -> bool:
        """Checks if the pathlet should end"""
        toNodeObj = self.graph.nodesDict[toNode]
        # Check if toNode is a sink
        if toNodeObj.nodeType == 1:
            return True
        # Check if toNode has multiple outgoing flow
        outgoingFlows = 0
        for edge in toNodeObj.outgoingEdges:
            if self.getEdgeFlow(edge) > 0:
                outgoingFlows += 1
        if outgoingFlows > 1:
            return True
        # Check if toNode has other incoming flow
        incomingFlows = 0
        for edge in toNodeObj.incomingEdges:
            if self.getEdgeFlow(edge) > 0:
                incomingFlows += 1
        if incomingFlows > 1:
            return True
        return False

    def isDuplicatePathlet(self, visitedNodes: list) -> bool:
        """Checks if a constructed pathlet is already present in the individual's path list"""
        thisNodeTuple = tuple(visitedNodes)
        existingPathSet = set()
        for path in self.paths:
            existingNodeTuple = tuple(path.nodes)
            existingPathSet.add(existingNodeTuple)
        if thisNodeTuple in existingPathSet:
            return True
        else:
            return False

    def getEdgeFlow(self, edge: tuple) -> float:
        """Calculates the combined flow of the edge by considering all parallel arcs"""
        edgeFlow = 0.0
        edgeIndex = self.graph.edgesDict[edge]
        for capIndex in range(self.graph.numArcsPerEdge):
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
            edgeIndex = self.graph.edgesDict[edge]
            for capIndex in range(self.graph.numArcsPerEdge):
                thisArcFlow = self.arcFlows[(edgeIndex, capIndex)]
                if thisArcFlow > 0:
                    cap = self.graph.possibleArcCapsArray[capIndex]
                    fixedCost = self.graph.arcsDict[(edge[0], edge[1], cap)].fixedCost
                    variableCost = self.graph.arcsDict[(edge[0], edge[1], cap)].variableCost
                    pathFixedCost += fixedCost
                    pathVariableCost += variableCost * thisArcFlow
        # Account for source/sink costs
        for node in visitedNodes:
            nodeObj = self.graph.nodesDict[node]
            if nodeObj.nodeType == 0:
                for srcIndex in range(self.graph.numSources):
                    if self.graph.sourcesArray[srcIndex] == node:
                        variableCost = self.graph.sourceVariableCostsArray[srcIndex]
                        sinkSrcCost += variableCost * pathFlow
            elif nodeObj.nodeType == 1:
                for sinkIndex in range(self.graph.numSinks):
                    if self.graph.sinksArray[sinkIndex] == node:
                        variableCost = self.graph.sinkVariableCostsArray[sinkIndex]
                        sinkSrcCost += variableCost * pathFlow
        return pathFlow, pathFixedCost, pathVariableCost, sinkSrcCost

    def printAllPaths(self) -> None:
        """Prints the data for all paths constructed"""
        if len(self.paths) == 0:
            self.computeAllUsedPaths()
        for path in self.paths:
            path.printPathData()
