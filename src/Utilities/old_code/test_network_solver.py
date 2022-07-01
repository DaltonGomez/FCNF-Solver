from src.FlowNetwork.CandidateGraph import CandidateGraph
from src.FlowNetwork.GraphVisualizer import GraphVisualizer
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 test_network_solver.py
"""

if __name__ == "__main__":
    name = ""
    flowNetwork = CandidateGraph()
    flowNetwork = flowNetwork.loadCandidateGraph(name)

    # FlowNetwork Visualization Test
    visualizer = GraphVisualizer(flowNetwork, directed=False, supers=False)
    visualizer.drawUnlabeledGraph()

    # Solver Test
    solver = MILPsolverCPLEX(flowNetwork, 1, isOneArcPerEdge=False)
    solver.buildModel()
    solver.solveModel()
    solver.printAllSolverData()

    # Solution Test
    solution = solver.writeSolution()
    # solution.saveSolution()

    # Solution Visualizer Test
    solnVisualizer = SolutionVisualizer(solution)
    # solnVisualizer.drawUnlabeledGraph()
    solnVisualizer.drawLabeledSolution()
