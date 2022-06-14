import math
import random
from typing import List, Dict, Tuple

import numpy as np
from scipy.spatial import Delaunay

from src.FlowNetwork.CandidateGraph import CandidateGraph


class GraphGenerator:
    """Class that generates pseudorandom candidate graphs from clustered sources/sinks and their Delaunay triangulation"""

    def __init__(self, name: str, numNodes: int, numSources: int, minSourceClusters: int, sourcesPerClusterRange: tuple,
                 numSinks: int, minSinkClusters: int, sinksPerClusterRange: tuple, clusterRadiusRange: tuple):
        """Constructor of a GraphGenerator instance"""
        # Hyperparameters for candidate graph generation/computing pseudo-random costs
        self.embeddingSize: float = 100.0  # The n x n size in R^2 that the candidate graph is embedded in
        self.arcCostLookupTable: List[List[float, float, float]] = self.getArcCostLookupTable()  # List of all posible arc capacities, their fixed cost scalar, and their variable cost scalar
        self.edgePenaltyRange: List[float, float] = [0.95, 1.50]  # Range that is uniformly, randomly sampled for assigning edge penalties
        self.randomEdgePenalties: Dict[Tuple[int, int], float] = {}  # Dictionary mapping edgeID keys, given as (fromNode, toNode), to values of the edge penalty assigned
        self.isSourceSinkCapacitated: bool = True  # Boolean flag indicating if sources and sinks are capacitated
        self.sourceCapRange: List[float, float] = [1, 20]  # Range that is uniformly, randomly sampled for assigning source capacities
        self.sinkCapRange: List[float, float] = [1, 20]  # Range that is uniformly, randomly sampled for assigning sink capacities
        self.isSourceSinkCharged: bool = False  # Boolean flag indicating if sources and sinks are charged
        self.sourceChargeRange: List[float, float] = [1, 20]  # Range that is uniformly, randomly sampled for assigning source variable costs
        self.sinkChargeRange: List[float, float] = [1, 20]  # Range that is uniformly, randomly sampled for assigning sink variable costs
        # Cluster specific hyperparameters
        self.minSourceClusters: int = minSourceClusters  # Minimum number of source clusters that are attempted
        self.sourcesPerClusterRange: Tuple[int, int] = sourcesPerClusterRange  # Range that is uniformly, randomly sampled to determine the sources in each cluster
        self.minSinkClusters: int = minSinkClusters  # Minimum number of sink clusters that are attempted
        self.sinksPerClusterRange: Tuple[int, int] = sinksPerClusterRange  # Range that is uniformly, randomly sampled to determine the sinks in each cluster
        self.clusterRadiusRange: Tuple[int, int] = clusterRadiusRange  # Range that is uniformly, randomly sampled to determine the maximum radius of each cluster
        self.tempNodeIDs: List[Tuple[int, float, float]] = []  # Temporary data structure to hold node IDs before casting to an numpy array
        self.tempPoints: List[Tuple[float, float]] = []  # Temporary data structure to hold node (x-position, y-position) before casting to an numpy array

        # Output candidate graph to be built
        self.newGraph: CandidateGraph = CandidateGraph()  # Candidate graph object that the graph generator object constructs
        self.newGraph.name = name
        self.newGraph.numTotalNodes = numNodes
        self.newGraph.numSources = numSources
        self.newGraph.numSinks = numSinks

    def generateGraph(self) -> CandidateGraph:
        """Constructs a pseudo-random candidate graph embedded in a 2D plane"""
        self.embedRandomIntermediatePoints()
        self.assignClusteredSourceSinks()
        self.buildEdgesFromTriangulation()
        self.computeEdgeDistances()
        self.randomEdgePenalties = self.initializeRandomEdgePenalties()
        self.assignInOutEdgesToNodes()
        self.setPossibleArcCapacities()
        self.buildArcsDictAndMatrix()
        self.assignSourceSinkCapAndCharge()
        return self.newGraph

    def embedRandomIntermediatePoints(self) -> None:
        """Randomly embeds the intermediate points (i.e. n - (s + t)) in a 2D plane"""
        random.seed()
        tempInterNodes = []
        for n in range(self.newGraph.numTotalNodes - (self.newGraph.numSources + self.newGraph.numSinks)):
            xPos = random.random() * self.embeddingSize
            yPos = random.random() * self.embeddingSize
            self.tempNodeIDs.append((n, xPos, yPos))
            self.newGraph.addNodeToDict(n, xPos, yPos)
            self.tempPoints.append((xPos, yPos))
            tempInterNodes.append(n)
        self.newGraph.interNodesArray = np.array(tempInterNodes, dtype='i')
        self.newGraph.numInterNodes = len(self.newGraph.interNodesArray)

    def assignClusteredSourceSinks(self) -> None:
        """Randomly embeds source and sink using a pseudo-random clustered strategy"""
        random.seed()
        # Create all sources in clusters
        tempSrcIDs = []
        remainingSources = self.newGraph.numSources
        # Create each cluster
        for srcCluster in range(self.minSourceClusters):
            # Randomly generate a number of sources for this cluster within the bounds
            thisClusterDensity = random.randint(self.sourcesPerClusterRange[0], self.sourcesPerClusterRange[1])
            # Check if a cluster can be filled; otherwise, give it remaining sources
            if remainingSources < thisClusterDensity:
                thisClusterOfSources = self.buildClusteredPoints(remainingSources)
            else:
                thisClusterOfSources = self.buildClusteredPoints(thisClusterDensity)
            # Decrement sources left to assign
            remainingSources -= thisClusterDensity
            # Create each source
            for sourcePos in thisClusterOfSources:
                nodeID = len(self.tempNodeIDs)
                tempSrcIDs.append(nodeID)
                self.tempNodeIDs.append((nodeID, sourcePos[0], sourcePos[1]))
                self.tempPoints.append((sourcePos[0], sourcePos[1]))
                self.newGraph.addNodeToDict(nodeID, sourcePos[0], sourcePos[1])
                self.newGraph.setNodeType(nodeID, 0)
        # Ensure all sources were assigned; otherwise assign randomly across R^2
        while len(tempSrcIDs) < self.newGraph.numSources:
            xPos = random.random() * self.embeddingSize
            yPos = random.random() * self.embeddingSize
            nodeID = len(self.tempNodeIDs)
            tempSrcIDs.append(nodeID)
            self.tempNodeIDs.append((nodeID, xPos, yPos))
            self.tempPoints.append((xPos, yPos))
            self.newGraph.addNodeToDict(nodeID, xPos, yPos)
            self.newGraph.setNodeType(nodeID, 0)
        # Build sources array
        self.newGraph.sourcesArray = np.array(tempSrcIDs, dtype='i')
        # Create all sinks in clusters
        tempSinkIDs = []
        remainingSinks = self.newGraph.numSinks
        # Create each cluster
        for sinkCluster in range(self.minSinkClusters):
            # Randomly generate a number of sources for this cluster within the bounds
            thisClusterDensity = random.randint(self.sinksPerClusterRange[0], self.sinksPerClusterRange[1])
            # Check if a cluster can be filled; otherwise, give it remaining sources
            if remainingSinks < thisClusterDensity:
                thisClusterOfSinks = self.buildClusteredPoints(remainingSinks)
            else:
                thisClusterOfSinks = self.buildClusteredPoints(thisClusterDensity)
            # Decrement sources left to assign
            remainingSinks -= thisClusterDensity
            # Create each source
            for sinkPos in thisClusterOfSinks:
                nodeID = len(self.tempNodeIDs)
                tempSinkIDs.append(nodeID)
                self.tempNodeIDs.append((nodeID, sinkPos[0], sinkPos[1]))
                self.tempPoints.append((sinkPos[0], sinkPos[1]))
                self.newGraph.addNodeToDict(nodeID, sinkPos[0], sinkPos[1])
                self.newGraph.setNodeType(nodeID, 1)
        # Ensure all sinks were assigned; otherwise assign randomly across R^2
        while len(tempSinkIDs) < self.newGraph.numSinks:
            xPos = random.random() * self.embeddingSize
            yPos = random.random() * self.embeddingSize
            nodeID = len(self.tempNodeIDs)
            tempSinkIDs.append(nodeID)
            self.tempNodeIDs.append((nodeID, xPos, yPos))
            self.tempPoints.append((xPos, yPos))
            self.newGraph.addNodeToDict(nodeID, xPos, yPos)
            self.newGraph.setNodeType(nodeID, 1)
        # Build sources array
        self.newGraph.sinksArray = np.array(tempSinkIDs, dtype='i')
        # Cast temp points list to ndArrays in the newGraph object
        self.newGraph.points = np.array(self.tempPoints, dtype='f')

    def buildClusteredPoints(self, numNodesInCluster: int) -> list:
        """Builds a cluster of points for source/sink generation sources or sinks"""
        random.seed()
        clusteredPoints = []
        # Randomly choose cluster radius
        thisClusterRadius = random.randint(self.clusterRadiusRange[0], self.clusterRadiusRange[1])
        # Build cluster center
        xClusterCenter = random.random() * self.embeddingSize
        yClusterCenter = random.random() * self.embeddingSize
        # Generate each point around center
        for thisPoint in range(numNodesInCluster):
            distFromCenter = thisClusterRadius * math.sqrt(random.random())
            angle = 2 * math.pi * random.random()
            xPos = xClusterCenter + distFromCenter * math.cos(angle)
            yPos = yClusterCenter + distFromCenter * math.sin(angle)
            clusteredPoints.append((xPos, yPos))
        return clusteredPoints

    def buildEdgesFromTriangulation(self) -> None:
        """Builds a Delaunay triangulation to determine edges"""
        triangulation = Delaunay(self.newGraph.points)  # Compute the Delaunay triangulation
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
        self.newGraph.edgesArray = np.array(list(edgeSet), dtype='i')
        self.newGraph.numEdges = len(self.newGraph.edgesArray)
        # Build edge dict
        for i in range(self.newGraph.numEdges):
            fromNode = self.newGraph.edgesArray[i][0]
            toNode = self.newGraph.edgesArray[i][1]
            self.newGraph.addEdgeToDict((fromNode, toNode), i)

    def computeEdgeDistances(self) -> None:
        """Calculates the Euclidean distance of each edge"""
        tempDistances = []
        for edge in self.newGraph.edgesArray:
            pointOne = self.newGraph.getNodeCoordinates(edge[0])
            pointTwo = self.newGraph.getNodeCoordinates(edge[1])
            distance = math.sqrt((pointTwo[0] - pointOne[0]) ** 2 + (pointTwo[1] - pointOne[1]) ** 2)
            tempDistances.append(distance)
        self.newGraph.distancesArray = np.array(tempDistances, dtype='f')

    def assignInOutEdgesToNodes(self) -> None:
        """Updates the topology of the graph at the node level by assigning ingoing and outgoing edges"""
        for edge in self.newGraph.edgesArray:
            thisEdge = (edge[0], edge[1])
            self.newGraph.addOutgoingEdgeToNode(edge[0], thisEdge)
            self.newGraph.addIncomingEdgeToNode(edge[1], thisEdge)

    def initializeRandomEdgePenalties(self) -> dict:
        """Initializes a random penalties for each opposing edge pair to mimic spatial variability in costs (i.e. a cost raster)"""
        random.seed()
        edgePenalties = {}
        for edge in self.newGraph.edgesArray:
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
        self.newGraph.possibleArcCapsArray = np.array(tempArcCaps, dtype='f')
        self.newGraph.numArcsPerEdge = len(self.newGraph.possibleArcCapsArray)

    def buildArcsDictAndMatrix(self) -> None:
        """Builds the dictionary of arcs"""
        tempArcs = []
        numID = 0
        for edgeID in range(self.newGraph.numEdges):
            fromNode = self.newGraph.edgesArray[edgeID][0]
            toNode = self.newGraph.edgesArray[edgeID][1]
            distance = self.newGraph.distancesArray[edgeID]
            for capID in range(self.newGraph.numArcsPerEdge):
                cap = self.newGraph.possibleArcCapsArray[capID]
                fixedCost = self.calculateArcFixedCost(edgeID, capID)
                variableCost = self.calculateArcVariableCost(edgeID, capID)
                self.newGraph.addArcToDict(numID, (fromNode, toNode, cap), distance, fixedCost, variableCost)
                thisArc = [numID, fromNode, toNode, cap, distance, fixedCost, variableCost]
                tempArcs.append(thisArc)
                numID += 1
        self.newGraph.arcsMatrix = np.array(tempArcs, dtype='f')
        self.newGraph.numTotalArcs = len(self.newGraph.arcsMatrix)

    def calculateArcFixedCost(self, edgeID: int, capID: int) -> float:
        """Calculates the fixed cost of the arc in a pseudorandom manner"""
        distance = self.newGraph.distancesArray[edgeID]
        thisEdge = self.newGraph.edgesArray[edgeID]
        penalty = self.randomEdgePenalties[(thisEdge[0], thisEdge[1])]
        fixedCostScalar = self.arcCostLookupTable[capID][1]
        fixedCost = distance * penalty * fixedCostScalar
        return fixedCost

    def calculateArcVariableCost(self, edgeID: int, capID: int) -> float:
        """Calculates the variable cost of the arc in a pseudorandom manner"""
        distance = self.newGraph.distancesArray[edgeID]
        thisEdge = self.newGraph.edgesArray[edgeID]
        penalty = self.randomEdgePenalties[(thisEdge[0], thisEdge[1])]
        variableCostScalar = self.arcCostLookupTable[capID][2]
        variableCost = distance * penalty * variableCostScalar
        return variableCost

    def assignSourceSinkCapAndCharge(self) -> None:
        """Assigns source/sink capacities and/or charges if requested"""
        random.seed()
        if self.isSourceSinkCapacitated is True:
            self.newGraph.isSourceSinkCapacitated = True
            tempSrcCaps = []
            for source in range(self.newGraph.numSources):
                thisSrcCap = random.uniform(self.sourceCapRange[0], self.sourceCapRange[1])
                tempSrcCaps.append(thisSrcCap)
            self.newGraph.sourceCapsArray = np.array(tempSrcCaps, dtype='f')
            tempSinkCaps = []
            for sink in range(self.newGraph.numSinks):
                thisSinkCap = random.uniform(self.sinkCapRange[0], self.sinkCapRange[1])
                tempSinkCaps.append(thisSinkCap)
            self.newGraph.sinkCapsArray = np.array(tempSinkCaps, dtype='f')
        if self.isSourceSinkCharged is True:
            self.newGraph.isSourceSinkCharged = True
            tempSrcCosts = []
            for source in range(self.newGraph.numSources):
                thisSrcCost = random.uniform(self.sourceChargeRange[0], self.sourceChargeRange[1])
                tempSrcCosts.append(thisSrcCost)
            self.newGraph.sourceVariableCostsArray = np.array(tempSrcCosts, dtype='f')
            tempSinkCosts = []
            for sink in range(self.newGraph.numSinks):
                thisSinkCost = random.uniform(self.sinkChargeRange[0], self.sinkChargeRange[1])
                tempSinkCosts.append(thisSinkCost)
            self.newGraph.sinkVariableCostsArray = np.array(tempSinkCosts, dtype='f')
        self.newGraph.totalPossibleDemand = self.newGraph.calculateTotalPossibleDemand()

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
