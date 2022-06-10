from Network.ClusteredGraphMaker import ClusteredGraphMaker

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 test_clustered_network_maker.py
"""

# ORIGINAL COST DETERMINING TABLE ADAPTED FROM SIMCCS MODEL (see data/SimCCS_models/Pipeline Cost Model.xlsm)
# Format: cap, fixed cost, variable cost
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

srcSinkCapacityRange = (0.01, 2)  # TODO - Determine range based on real datasets
srcSinkChargeRange = (0.01, 2)  # TODO - Determine range based on real datasets

if __name__ == "__main__":
    for n in range(10):
        # Input parameters
        name = "cluster_test_" + str(n)
        numNodes = 100
        numSources = 15
        numSinks = 15
        # Cluster parameters
        minSourceClusters = 3
        sourcesPerClusterRange = (3, 6)
        minSinkClusters = 3
        sinksPerClusterRange = (3, 6)
        clusterRadiusRange = (10, 20)
        # Source/sink cap ranges
        srcCapRange = (1, 20)
        sinkCapRange = (1, 20)
        # Make graph
        graphMaker = ClusteredGraphMaker(name, numNodes, numSources, minSourceClusters, sourcesPerClusterRange,
                                         numSinks, minSinkClusters, sinksPerClusterRange, clusterRadiusRange)
        graphMaker.setSourceSinkGeneralizations(isCapacitated=True, isCharged=False,
                                                srcCapRange=srcCapRange, sinkCapRange=sinkCapRange)
        newGraph = graphMaker.generateNetwork()
        newGraph.drawNetworkTriangulation()
        newGraph.saveNetwork()
