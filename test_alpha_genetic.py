from src.AlphaGenetic.Population import Population
from src.Network.FlowNetwork import FlowNetwork
from src.Network.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 test_alpha_genetic.py
"""

if __name__ == "__main__":
    network = FlowNetwork()
    network = network.loadNetwork("25-1-1.p")
    minTargetFlow = 100

    pop = Population(network, minTargetFlow, 10)
    pop.evolvePopulation(2, drawing=False)

    cplex = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
    cplex.buildModel()
    cplex.solveModel()
    opt = cplex.writeSolution()

    vis = SolutionVisualizer(opt)
    # vis.drawUnlabeledGraph(leadingText="OPT")
