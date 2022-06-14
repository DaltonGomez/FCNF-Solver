import math
import random

from Utilities.old_code.OLDGraphMaker import GraphMaker
from src.AntColony.AntColonyResults import AntColonyResults

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_ant_results_batch.py
"""

if __name__ == "__main__":
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
        # Arc Cost Determining Table
        # Format: cap, fixed cost, variable cost
        arcCostLookupTable = [
            [100, 10, 1]
        ]
        # Source/sink cap and cost ranges
        srcSinkCapacityRange = [100, 200]
        srcSinkChargeRange = [10, 25]
        graphMaker = GraphMaker(networkName, numNodes, numSrcSinks, numSrcSinks)
        graphMaker.setArcCostLookupTable(arcCostLookupTable=arcCostLookupTable)
        graphMaker.setSourceSinkGeneralizations(isCapacitated=True, isCharged=False,
                                                srcCapRange=(1, 20), sinkCapRange=(1, 20))
        generatedNetwork = graphMaker.generateNetwork()
        generatedNetwork.saveCandidateGraph()
        networkList.append(networkName)

    # Solve all networks
    numAnts = 50
    numEpisodes = 15
    experiment = AntColonyResults(networkList, numAnts, numEpisodes)
    experiment.runExperiment()
