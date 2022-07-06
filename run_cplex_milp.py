from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_cplex_milp.py
"""

if __name__ == "__main__":
    # Load FlowNetwork
    graph = CandidateGraph()
    graph = graph.loadCandidateGraph("test_8.p")
    minTargetFlow = graph.totalPossibleDemand

    # Draw input graph
    # vis = GraphVisualizer(graph, directed=True, supers=False)
    # vis.drawBidirectionalGraphWithSmoothedLabeledEdges()

    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(graph, minTargetFlow, logOutput=True)
    # cplex.setTimeLimit(10)
    opt = cplex.findSolution()

    # Write and draw solution
    optVis = SolutionVisualizer(opt)
    optVis.drawLabeledSolution(leadingText="OPT_")
