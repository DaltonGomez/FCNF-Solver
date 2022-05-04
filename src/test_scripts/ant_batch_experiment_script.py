import math
import random

from src.AntExperiments.AntResultsExperiment import AntResultsExperiment
from src.Network.GraphMaker import GraphMaker

numGraphs = 50
nodeSizeRange = [25, 400]
srcSinkSet = [1, 5, 10]
possibleArcCaps = [100]
networkList = []

# Automatically generate n input networks
for n in range(numGraphs):
    random.seed()
    # Uniformly sample number of nodes
    numNodes = random.randint(nodeSizeRange[0], nodeSizeRange[1])
    numSrcSinks = random.sample(srcSinkSet, 1)[0]
    # Keep sampling until there is enough nodes to support the sources and sinks
    while numSrcSinks > math.floor(numNodes / 2):
        numNodes = random.randint(nodeSizeRange[0], nodeSizeRange[1])
        numSrcSinks = random.randint(srcSinkSet[0], srcSinkSet[1])
    # Build and save the network
    networkName = str(numNodes) + "-" + str(len(possibleArcCaps)) + "-" + str(numSrcSinks) + "-" + str(n)
    graphMaker = GraphMaker(networkName, numNodes, numSrcSinks, numSrcSinks)
    graphMaker.setCostDeterminingHyperparameters(possibleArcCaps=possibleArcCaps)
    graphMaker.setSourceSinkGeneralizations(True, True)
    generatedNetwork = graphMaker.generateNetwork()
    generatedNetwork.saveNetwork()
    networkList.append(networkName)

# Solve all networks
numAnts = 50
numEpisodes = 15
experiment = AntResultsExperiment(networkList, numAnts, numEpisodes)
experiment.runExperiment()
