from src.Network.FlowNetwork import FlowNetwork
from src.Network.NetworkVisualizer import NetworkVisualizer
from src.Network.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 test_network_solver.py
"""

if __name__ == "__main__":
    name = "presEx3.p"
    flowNetwork = FlowNetwork()
    flowNetwork = flowNetwork.loadNetwork(name)

    # Network Visualization Test
    visualizer = NetworkVisualizer(flowNetwork, directed=False, supers=False)
    visualizer.drawUnlabeledGraph()

    # Solver Test
    solver = MILPsolverCPLEX(flowNetwork, 300, isOneArcPerEdge=False)
    solver.buildModel()
    solver.solveModel()
    solver.printAllSolverData()

    # Solution Test
    solution = solver.writeSolution()
    # solution.saveSolution()

    # Solution Visualizer Test
    solnVisualizer = SolutionVisualizer(solution)
    # solnVisualizer.drawUnlabeledGraph()
    solnVisualizer.drawGraphWithLabels()
