import math
import random
from typing import List

import numpy as np
from scipy.spatial import Delaunay

from src.Network.FlowNetwork import FlowNetwork


class GraphMaker:
    """Class that generates pseudorandom graphs from 2D embedded points and their Delaunay triangulation"""

    def __init__(self, name: str, numNodes: int, numSources: int, numSinks: int):
        """Constructor of a GraphGenerator instance"""
        # Hyperparameters For Network Generation/Computing Pseudo-Random Costs
        self.embeddingSize = 100.0
        self.possibleArcCaps = [10, 50, 100]
        self.distFixCostScale = 10.0
        self.capFixCostScale = 10.0
        self.fixCostRandomScalar = [0.80, 1.20]
        self.distVariableCostScale = 3.0
        self.capVariableCostScale = 3.0
        self.variableCostRandomScalar = [0.80, 1.20]
        self.isSourceSinkCapacitated = True
        self.sourceSinkCapacityRange = [100, 300]
        self.isSourceSinkCharged = True
        self.sourceSinkChargeRange = [200.0, 300.0]

        # Cost Scalar Lookup Tables
        self.costScalarLookupTable = self.initializeCostLookupTable()

        # Output Network To Be Built
        self.newNetwork = FlowNetwork()
        self.newNetwork.name = name
        self.newNetwork.numTotalNodes = numNodes
        self.newNetwork.numSources = numSources
        self.newNetwork.numSinks = numSinks

    def generateNetwork(self) -> FlowNetwork:
        """Constructs a pseudo-random flow network embedded in a 2D plane"""
        self.embedRandomPoints()
        self.assignRandomSourceSinks()
        self.buildEdgesFromTriangulation()
        self.computeEdgeDistances()
        self.assignInOutEdgesToNodes()
        self.setPossibleArcCapacities(self.possibleArcCaps)
        self.buildArcsDictAndMatrix()
        self.assignSourceSinkCapAndCharge()
        return self.newNetwork

    def initializeCostLookupTable(self) -> dict:
        """Initializes the cost scalar lookup table (loosely based on the SimCCS model formulation)"""
        pass
        # TODO - Implement the new cost function discussed with Sean

    def setCostDeterminingHyperparameters(self, embeddingSize=100.0, possibleArcCaps=(10, 50, 100),
                                          distFixCostScale=10.0, capFixCostScale=10.0, fixCostRandomScalar=(0.80, 1.20),
                                          distVariableCostScale=3.0, capVariableCostScale=3.0,
                                          variableCostRandomScalar=(0.80, 1.20)) -> None:
        """Allows the hyperparameters that calculate the pseudorandom cost to be tuned"""
        self.embeddingSize = embeddingSize
        self.possibleArcCaps = possibleArcCaps
        self.distFixCostScale = distFixCostScale
        self.capFixCostScale = capFixCostScale
        self.fixCostRandomScalar = fixCostRandomScalar
        self.distVariableCostScale = distVariableCostScale
        self.capVariableCostScale = capVariableCostScale
        self.variableCostRandomScalar = variableCostRandomScalar

    def setSourceSinkGeneralizations(self, isCapacitated: bool, isCharged: bool, capacityRange=(100, 300),
                                     chargeRange=(200.0, 300.0)) -> None:
        """Allows the capacitated/charged source sink generalizes to be turned on and tuned"""
        self.isSourceSinkCapacitated = isCapacitated
        self.sourceSinkCapacityRange = capacityRange
        self.isSourceSinkCharged = isCharged
        self.sourceSinkChargeRange = chargeRange

    def embedRandomPoints(self) -> None:
        """Randomly embeds n points in a 2D plane"""
        random.seed()
        tempPoints = []
        for n in range(self.newNetwork.numTotalNodes):
            xPos = random.random() * self.embeddingSize
            yPos = random.random() * self.embeddingSize
            tempPoints.append((xPos, yPos))
            self.newNetwork.addNodeToDict(n, xPos, yPos)
        self.newNetwork.points = np.array(tempPoints)

    def assignRandomSourceSinks(self) -> None:
        """Randomly assigns source and sink IDs to nodes"""
        tempNodes = set(range(self.newNetwork.numTotalNodes))
        tempSrcSinks = set(random.sample(tempNodes, self.newNetwork.numSources + self.newNetwork.numSinks))
        tempInterNodes = tempNodes.symmetric_difference(tempSrcSinks)
        tempSources = set(random.sample(tempSrcSinks, self.newNetwork.numSources))
        tempSinks = tempSrcSinks.symmetric_difference(tempSources)
        self.newNetwork.sourcesArray = np.array(list(tempSources))
        for source in self.newNetwork.sourcesArray:
            self.newNetwork.setNodeType(source, 0)
        self.newNetwork.sinksArray = np.array(list(tempSinks))
        for sink in self.newNetwork.sinksArray:
            self.newNetwork.setNodeType(sink, 1)
        self.newNetwork.interNodesArray = np.array(list(tempInterNodes))
        self.newNetwork.numInterNodes = len(self.newNetwork.interNodesArray)

    def buildEdgesFromTriangulation(self) -> None:
        """Builds a Delaunay triangulation to determine edges"""
        triangulation = Delaunay(self.newNetwork.points)  # Compute the Delaunay triangulation
        edgeSet = set()  # Declare a new set for the edges
        for simplex in range(triangulation.nsimplex):  # Iterate over each simplex
            firstNode = -1  # To complete the third edge of the simplex
            for vertex in range(3):  # Iterate over the three points in a simplex
                if vertex == 0:
                    firstNode = triangulation.simplices[simplex, vertex]  # Store the first vertex in the simplex
                if vertex != 2:
                    edgeList = [triangulation.simplices[simplex, vertex],
                                triangulation.simplices[simplex, vertex + 1]]  # Makes bi-directional edge
                    forwardEdge = tuple(sorted(edgeList, reverse=False))  # Makes unidirectional forward edge
                    edgeSet.add(forwardEdge)  # Adds to set for deduplication
                    backwardEdge = tuple(sorted(edgeList, reverse=True))  # Makes unidirectional backward edge
                    edgeSet.add(backwardEdge)  # Adds to set for deduplication
                else:  # Logic for the edge connecting the last point to the first
                    edgeList = (triangulation.simplices[simplex, vertex], firstNode)
                    forwardEdge = tuple(sorted(edgeList, reverse=False))
                    edgeSet.add(forwardEdge)
                    backwardEdge = tuple(sorted(edgeList, reverse=True))
                    edgeSet.add(backwardEdge)
        self.newNetwork.edgesArray = np.array(list(edgeSet))
        self.newNetwork.numEdges = len(self.newNetwork.edgesArray)
        # Build edge dict
        for i in range(self.newNetwork.numEdges):
            fromNode = self.newNetwork.edgesArray[i][0]
            toNode = self.newNetwork.edgesArray[i][1]
            self.newNetwork.addEdgeToDict((fromNode, toNode), i)

    def computeEdgeDistances(self) -> None:
        """Calculates the Euclidean distance of each edge"""
        tempDistances = []
        for edge in self.newNetwork.edgesArray:
            pointOne = self.newNetwork.getNodeCoordinates(edge[0])
            pointTwo = self.newNetwork.getNodeCoordinates(edge[1])
            distance = math.sqrt((pointTwo[0] - pointOne[0]) ** 2 + (pointTwo[1] - pointOne[1]) ** 2)
            tempDistances.append(distance)
        self.newNetwork.distancesArray = np.array(tempDistances)

    def assignInOutEdgesToNodes(self) -> None:
        """Updates the topology of the network at the node level by assigning ingoing and outgoing edges"""
        for edge in self.newNetwork.edgesArray:
            thisEdge = (edge[0], edge[1])
            self.newNetwork.addOutgoingEdgeToNode(edge[0], thisEdge)
            self.newNetwork.addIncomingEdgeToNode(edge[1], thisEdge)

    def setPossibleArcCapacities(self, possibleArcCapacities: List[int]) -> None:
        """Sets the possible arc capacities for the parallel edges"""
        self.newNetwork.possibleArcCapsArray = np.array(possibleArcCapacities)
        self.newNetwork.numArcCaps = len(self.newNetwork.possibleArcCapsArray)

    def buildArcsDictAndMatrix(self) -> None:
        """Builds the dictionary of arcs"""
        tempArcs = []
        numID = 0
        for edge in range(self.newNetwork.numEdges):
            fromNode = self.newNetwork.edgesArray[edge][0]
            toNode = self.newNetwork.edgesArray[edge][1]
            distance = self.newNetwork.distancesArray[edge]
            for cap in self.newNetwork.possibleArcCapsArray:
                fixedCost = self.calculateArcFixedCost(distance, cap)
                variableCost = self.calculateArcVariableCost(distance, cap)
                self.newNetwork.addArcToDict(numID, (fromNode, toNode, cap), distance, fixedCost, variableCost)
                thisArc = [numID, fromNode, toNode, cap, distance, fixedCost, variableCost]
                tempArcs.append(thisArc)
                numID += 1
        self.newNetwork.arcsMatrix = np.array(tempArcs)
        self.newNetwork.numArcs = len(self.newNetwork.arcsMatrix)

    def calculateArcFixedCost(self, distance: float, capacity: int) -> float:
        """Calculates the fixed cost of the arc in a pseudorandom manner"""
        # TODO - Apply the pipeline cost function: c(f) = (m*cap + b) * edge_specific_penalty (which is based on distance)
        # Pseudorandom component proportional to the distance the edge spans
        randomDistanceComponent = (self.distFixCostScale * distance * random.uniform(
            self.fixCostRandomScalar[0], self.fixCostRandomScalar[1]))
        # Cap^(3/4) is intended to discount bigger pipelines (i.e. economies of scale)
        fixedCost = (randomDistanceComponent + self.capFixCostScale * capacity ** 0.75)
        return fixedCost

    def calculateArcVariableCost(self, distance: float, capacity: int) -> float:
        """Calculates the variable cost of the arc in a pseudorandom manner"""
        # Pseudorandom component proportional to the distance the edge spans
        randomDistanceComponent = (self.distVariableCostScale * distance * random.uniform(
            self.variableCostRandomScalar[0], self.variableCostRandomScalar[1]))
        # Cap^(3/4) is intended to discount bigger pipelines (i.e. economies of scale)
        variableCost = (randomDistanceComponent + self.capVariableCostScale * capacity ** 0.75)
        return variableCost

    def assignSourceSinkCapAndCharge(self) -> None:
        """Assigns source/sink capacities and/or charges if requested"""
        if self.isSourceSinkCapacitated is True:
            self.newNetwork.isSourceSinkCapacitated = True
            tempSrcCaps = []
            for source in range(self.newNetwork.numSources):
                thisSrcCap = random.randint(self.sourceSinkCapacityRange[0], self.sourceSinkCapacityRange[1])
                tempSrcCaps.append(thisSrcCap)
            self.newNetwork.sourceCapsArray = np.array(tempSrcCaps)
            tempSinkCaps = []
            for sink in range(self.newNetwork.numSinks):
                thisSinkCap = random.randint(self.sourceSinkCapacityRange[0], self.sourceSinkCapacityRange[1])
                tempSinkCaps.append(thisSinkCap)
            self.newNetwork.sinkCapsArray = np.array(tempSinkCaps)
        if self.isSourceSinkCharged is True:
            self.newNetwork.isSourceSinkCharged = True
            tempSrcCosts = []
            for source in range(self.newNetwork.numSources):
                thisSrcCost = random.uniform(self.sourceSinkChargeRange[0], self.sourceSinkChargeRange[1])
                tempSrcCosts.append(thisSrcCost)
            self.newNetwork.sourceVariableCostsArray = np.array(tempSrcCosts)
            tempSinkCosts = []
            for sink in range(self.newNetwork.numSinks):
                thisSinkCost = random.uniform(self.sourceSinkChargeRange[0], self.sourceSinkChargeRange[1])
                tempSinkCosts.append(thisSinkCost)
            self.newNetwork.sinkVariableCostsArray = np.array(tempSinkCosts)
