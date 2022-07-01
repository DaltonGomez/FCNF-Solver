from src.FlowNetwork.CandidateGraph import CandidateGraph
from src.FlowNetwork.GraphVisualizer import GraphVisualizer
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_cplex_milp.py
"""

if __name__ == "__main__":
    # Load FlowNetwork
    graph = CandidateGraph()
    graph = graph.loadCandidateGraph("huge_6.p")
    vis = GraphVisualizer(graph, directed=True, supers=False)
    vis.drawBidirectionalGraphWithSmoothedLabeledEdges()
    minTargetFlow = graph.totalPossibleDemand

    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(graph, minTargetFlow, isOneArcPerEdge=False)
    cplex.findSolution(printDetails=False)
    opt = cplex.writeSolution()
    optVis = SolutionVisualizer(opt)
    optVis.drawLabeledSolution(leadingText="OPT_")