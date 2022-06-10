from src.Network.FlowNetwork import FlowNetwork
from src.Network.NetworkVisualizer import NetworkVisualizer
from src.Network.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_cplex_milp.py
"""

if __name__ == "__main__":
    # Load Network
    network = FlowNetwork()
    network = network.loadNetwork("basic_5.p")
    vis = NetworkVisualizer(network, directed=True, supers=False)
    # vis.drawBidirectionalGraphWithSmoothedLabeledEdges()
    minTargetFlow = network.totalPossibleDemand

    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
    cplex.findSolution(printDetails=True)
    opt = cplex.writeSolution()
    optVis = SolutionVisualizer(opt)
    optVis.drawGraphWithLabels(leadingText="OPT_")
