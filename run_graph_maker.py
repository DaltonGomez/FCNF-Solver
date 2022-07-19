from src.Graph.GraphGenerator import GraphGenerator

"""
WINDOWS RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_graph_maker.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 run_graph_maker.py
"""

# ORIGINAL COST DETERMINING TABLE ADAPTED FROM SIMCCS MODEL (see data/models/PipelineCostModel.xlsm)
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

if __name__ == "__main__":
    numGraphs = 10
    for n in range(numGraphs):
        # Input parameters
        name = "massive_" + str(n)
        numNodes = 500
        numSources = 50
        numSinks = 50
        # Cluster parameters
        minSourceClusters = 4
        sourcesPerClusterRange = (7, 14)
        minSinkClusters = 4
        sinksPerClusterRange = (7, 14)
        clusterRadiusRange = (10, 20)
        # Source/sink cap ranges
        isSrcSinkCapacitated = True
        srcCapRange = (5, 20)
        sinkCapRange = (5, 20)
        # Make graph
        graphMaker = GraphGenerator(name, numNodes, numSources, minSourceClusters, sourcesPerClusterRange,
                                    numSinks, minSinkClusters, sinksPerClusterRange, clusterRadiusRange)
        graphMaker.setSourceSinkGeneralizations(isCapacitated=isSrcSinkCapacitated, isCharged=False,
                                                srcCapRange=srcCapRange, sinkCapRange=sinkCapRange)
        newGraph = graphMaker.generateGraph()
        newGraph.drawGraphTriangulation()
        newGraph.saveCandidateGraph()
