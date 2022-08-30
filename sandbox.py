from FlowNetwork.SolutionVisualizer import SolutionVisualizer
from Graph.CandidateGraph import CandidateGraph
from Graph.GraphVisualizer import GraphVisualizer
from Solvers.MILPsolverCPLEX import MILPsolverCPLEX
from TransportationReduction.TransportationProblem import TransportationProblem

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 sandbox.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 sandbox.py
"""

if __name__ == "__main__":
    print("Run sandbox code here...")
    """
    # Graph generator code
    costLookupTable = [
        [10, 10, 10]
    ]
    name = "???"
    numNodes = 6
    numSources = 2
    numSinks = 2
    # Cluster parameters
    minSourceClusters = 2
    sourcesPerClusterRange = (2, 2)
    minSinkClusters = 2
    sinksPerClusterRange = (2, 2)
    clusterRadiusRange = (10, 15)
    # Source/sink cap ranges
    isSrcSinkCapacitated = True
    srcCapRange = (10, 10)
    sinkCapRange = (10, 10)
    # Make graph
    graphMaker = GraphGenerator(name, numNodes, numSources, minSourceClusters, sourcesPerClusterRange,
                                numSinks, minSinkClusters, sinksPerClusterRange, clusterRadiusRange)
    graphMaker.setSourceSinkGeneralizations(isCapacitated=isSrcSinkCapacitated, isCharged=False,
                                            srcCapRange=srcCapRange, sinkCapRange=sinkCapRange)
    graphMaker.setArcCostLookupTable(arcCostLookupTable=costLookupTable)
    newGraph = graphMaker.generateGraph()
    newGraph.drawGraphTriangulation()
    newGraph.saveCandidateGraph()
    """

    # Load and draw graph
    graphFile = "FCTP_TEST2.p"
    inputGraph = CandidateGraph()
    inputGraph = inputGraph.loadCandidateGraph(graphFile)
    minTargetFlow = inputGraph.calculateTotalPossibleDemand()
    graphVis = GraphVisualizer(inputGraph, directed=True, supers=False)
    graphVis.drawLabeledGraph()
    # Perform reduction
    fctp = TransportationProblem(inputGraph, minTargetFlow)

    # Solve optimally
    milpSolver = MILPsolverCPLEX(inputGraph, minTargetFlow, isOneArcPerEdge=True, logOutput=True)
    milpSolution = milpSolver.findSolution()
    solVis = SolutionVisualizer(milpSolution)
    solVis.drawLabeledSolution(leadingText="MILP")

