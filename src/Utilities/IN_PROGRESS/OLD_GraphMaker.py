import math
import random

import numpy as np
from scipy.spatial import Delaunay

from Graph.CandidateGraph import CandidateGraph


class GraphMaker:
    """Class that generates pseudorandom graphs from 2D embedded points and their Delaunay triangulation"""

    def __init__(self, name: str, numNodes: int, numSources: int, numSinks: int):
        """Constructor of a GraphMaker instance"""
        # Hyperparameters For FlowNetwork Generation/Computing Pseudo-Random Costs
        self.embeddingSize = 100.0
        self.arcCostLookupTable = self.getArcCostLookupTable()
        self.edgePenaltyRange = [0.95, 1.50]
        self.randomEdgePenalties = None
        self.isSourceSinkCapacitated = True
        self.sourceCapRange = [1, 20]
        self.sinkCapRange = [1, 20]
        self.isSourceSinkCharged = False
        self.sourceChargeRange = [1, 20]
        self.sinkChargeRange = [1, 20]

        # Output FlowNetwork To Be Built
        self.newNetwork = CandidateGraph()
        self.newNetwork.name = name
        self.newNetwork.numTotalNodes = numNodes
        self.newNetwork.numSources = numSources
        self.newNetwork.numSinks = numSinks

    def generateNetwork(self) -> CandidateGraph:
        """Constructs a pseudo-random flow network embedded in a 2D plane"""
        self.embedRandomPoints()
        self.assignRandomSourceSinks()
        self.buildEdgesFromTriangulation()
        self.computeEdgeDistances()
        self.randomEdgePenalties = self.initializeRandomEdgePenalties()
        self.assignInOutEdgesToNodes()
        self.setPossibleArcCapacities()
        self.buildArcsDictAndMatrix()
        self.assignSourceSinkCapAndCharge()
        return self.newNetwork

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
        random.seed()
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

    def initializeRandomEdgePenalties(self) -> dict:
        """Initializes a random penalties for each opposing edge pair to mimic spatial variability in costs (i.e. the cost raster)"""
        random.seed()
        edgePenalties = {}
        for edge in self.newNetwork.edgesArray:
            thisEdge = (edge[0], edge[1])
            backEdge = (edge[1], edge[0])
            thisPenalty = random.uniform(self.edgePenaltyRange[0], self.edgePenaltyRange[1])
            edgePenalties[thisEdge] = thisPenalty
            edgePenalties[backEdge] = thisPenalty
        return edgePenalties

    def setPossibleArcCapacities(self) -> None:
        """Sets the possible arc capacities for the parallel edges"""
        tempArcCaps = []
        for cap in self.arcCostLookupTable:
            tempArcCaps.append(cap[0])
        self.newNetwork.possibleArcCapsArray = np.array(tempArcCaps)
        self.newNetwork.numArcsPerEdge = len(self.newNetwork.possibleArcCapsArray)

    def buildArcsDictAndMatrix(self) -> None:
        """Builds the dictionary of arcs"""
        tempArcs = []
        numID = 0
        for edgeID in range(self.newNetwork.numEdges):
            fromNode = self.newNetwork.edgesArray[edgeID][0]
            toNode = self.newNetwork.edgesArray[edgeID][1]
            distance = self.newNetwork.distancesArray[edgeID]
            for capID in range(self.newNetwork.numArcsPerEdge):
                cap = self.newNetwork.possibleArcCapsArray[capID]
                fixedCost = self.calculateArcFixedCost(edgeID, capID)
                variableCost = self.calculateArcVariableCost(edgeID, capID)
                self.newNetwork.addArcToDict(numID, (fromNode, toNode, cap), distance, fixedCost, variableCost)
                thisArc = [numID, fromNode, toNode, cap, distance, fixedCost, variableCost]
                tempArcs.append(thisArc)
                numID += 1
        self.newNetwork.arcsMatrix = np.array(tempArcs)
        self.newNetwork.numArcs = len(self.newNetwork.arcsMatrix)

    def calculateArcFixedCost(self, edgeID: int, capID: int) -> float:
        """Calculates the fixed cost of the arc in a pseudorandom manner"""
        distance = self.newNetwork.distancesArray[edgeID]
        thisEdge = self.newNetwork.edgesArray[edgeID]
        penalty = self.randomEdgePenalties[(thisEdge[0], thisEdge[1])]
        fixedCostScalar = self.arcCostLookupTable[capID][1]
        fixedCost = distance * penalty * fixedCostScalar
        return fixedCost

    def calculateArcVariableCost(self, edgeID: int, capID: int) -> float:
        """Calculates the variable cost of the arc in a pseudorandom manner"""
        distance = self.newNetwork.distancesArray[edgeID]
        thisEdge = self.newNetwork.edgesArray[edgeID]
        penalty = self.randomEdgePenalties[(thisEdge[0], thisEdge[1])]
        variableCostScalar = self.arcCostLookupTable[capID][2]
        variableCost = distance * penalty * variableCostScalar
        return variableCost

    def assignSourceSinkCapAndCharge(self) -> None:
        """Assigns source/sink capacities and/or charges if requested"""
        random.seed()
        if self.isSourceSinkCapacitated is True:
            self.newNetwork.isSourceSinkCapacitated = True
            tempSrcCaps = []
            for source in range(self.newNetwork.numSources):
                thisSrcCap = random.uniform(self.sourceCapRange[0], self.sourceCapRange[1])
                tempSrcCaps.append(thisSrcCap)
            self.newNetwork.sourceCapsArray = np.array(tempSrcCaps)
            tempSinkCaps = []
            for sink in range(self.newNetwork.numSinks):
                thisSinkCap = random.uniform(self.sinkCapRange[0], self.sinkCapRange[1])
                tempSinkCaps.append(thisSinkCap)
            self.newNetwork.sinkCapsArray = np.array(tempSinkCaps)
        if self.isSourceSinkCharged is True:
            self.newNetwork.isSourceSinkCharged = True
            tempSrcCosts = []
            for source in range(self.newNetwork.numSources):
                thisSrcCost = random.uniform(self.sourceChargeRange[0], self.sourceChargeRange[1])
                tempSrcCosts.append(thisSrcCost)
            self.newNetwork.sourceVariableCostsArray = np.array(tempSrcCosts)
            tempSinkCosts = []
            for sink in range(self.newNetwork.numSinks):
                thisSinkCost = random.uniform(self.sinkChargeRange[0], self.sinkChargeRange[1])
                tempSinkCosts.append(thisSinkCost)
            self.newNetwork.sinkVariableCostsArray = np.array(tempSinkCosts)
        self.newNetwork.totalPossibleDemand = self.newNetwork.calculateTotalPossibleDemand()

    def setArcCostLookupTable(self, embeddingSize=100.0, edgePenaltyRange=(0.95, 1.50),
                              arcCostLookupTable=(
                                      [0.19, 12.38148027, 10.68565602],
                                      [0.54, 14.09940018, 4.500588868],
                                      [1.13, 16.21640614, 2.747501568],
                                      [3.25, 21.69529036, 1.700860451],
                                      [6.86, 30.97486282, 1.407282483],
                                      [12.26, 41.79573329, 1.2908691],
                                      [19.69, 55.47324885, 1.235063683],
                                      [35.13, 77.6425424, 1.194592382],
                                      [56.46, 104.7159663, 1.175094135],
                                      [83.95, 136.7519562, 1.164578466],
                                      [119.16, 172.7476864, 1.09878848],
                              )) -> None:
        """Allows the hyperparameters that calculate the pseudorandom cost to be tuned"""
        self.embeddingSize = embeddingSize
        self.edgePenaltyRange = edgePenaltyRange
        self.arcCostLookupTable = arcCostLookupTable

    def setSourceSinkGeneralizations(self, isCapacitated=True, isCharged=False, srcCapRange=(1, 20),
                                     sinkCapRange=(1, 20),
                                     srcChargeRange=(1, 20), sinkChargeRange=(1, 20)) -> None:
        """Allows the capacitated/charged source/sink generalizations to be turned on and tuned"""
        self.isSourceSinkCapacitated = isCapacitated
        self.sourceCapRange = srcCapRange
        self.sinkCapRange = sinkCapRange
        self.isSourceSinkCharged = isCharged
        self.sourceChargeRange = srcChargeRange
        self.sinkChargeRange = sinkChargeRange

    @staticmethod
    def getArcCostLookupTable() -> list:
        """Initializes the cost scalar lookup table (loosely based on the SimCCS model formulation- See data dir)"""
        # Columns: capacity, fixed cost, variable cost
        # DIRECTLY FROM THE SIMCCS PIPELINE MODEL (SCALED BY 1000)
        costLookupTable = [
            [0.19, 12.38148027, 10.68565602],
            [0.54, 14.09940018, 4.500588868],
            [1.13, 16.21640614, 2.747501568],
            [3.25, 21.69529036, 1.700860451],
            [6.86, 30.97486282, 1.407282483],
            [12.26, 41.79573329, 1.2908691],
            [19.69, 55.47324885, 1.235063683],
            [35.13, 77.6425424, 1.194592382],
            [56.46, 104.7159663, 1.175094135],
            [83.95, 136.7519562, 1.164578466],
            [119.16, 172.7476864, 1.09878848],
        ]
        return costLookupTable
