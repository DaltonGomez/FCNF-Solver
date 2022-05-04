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
    network = network.loadNetwork("1000-1-10.p")
    minTargetFlow = 1000

    pop = Population(network, minTargetFlow)
    pop.evolvePopulation(drawing=True, drawLabels=True)

    cplex = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
    cplex.buildModel()
    cplex.solveModel()
    opt = cplex.writeSolution()

    vis = SolutionVisualizer(opt)
    vis.drawGraphWithLabels(leadingText="OPT")
