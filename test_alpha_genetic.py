from src.AlphaGenetic.Population import Population
from src.Network.FlowNetwork import FlowNetwork
from src.Network.NetworkVisualizer import NetworkVisualizer
from src.Network.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 test_alpha_genetic.py
"""

if __name__ == "__main__":
    # Load Network
    network = FlowNetwork()
    network = network.loadNetwork("test2.p")
    vis = NetworkVisualizer(network, directed=True, supers=False)
    vis.drawBidirectionalGraphWithSmoothedLabeledEdges()
    minTargetFlow = 200

    # Solve with Alpha-GA
    pop = Population(network, minTargetFlow, populationSize=10, numGenerations=4)
    pop.setIndividualSelectionHyperparams("tournament", 3)
    pop.evolvePopulation(drawing=True, drawLabels=True)

    # Solve with Naive Hill Climb
    # pop.solveWithNaiveHillClimb(drawing=True, drawLabels=True)

    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
    cplex.findSolution(printDetails=False)
    opt = cplex.writeSolution()
    optVis = SolutionVisualizer(opt)
    optVis.drawGraphWithLabels(leadingText="OPT_")
