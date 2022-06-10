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
    # Load Network
    network = FlowNetwork()
    network = network.loadNetwork("basic_5.p")
    # vis = NetworkVisualizer(network, directed=True, supers=False)
    # vis.drawBidirectionalGraphWithSmoothedLabeledEdges()
    minTargetFlow = network.totalPossibleDemand

    # Initialize an Alpha-GA Population
    pop = Population(network, minTargetFlow)

    """
    # Solve with Naive Hill Climb
    hillClimb = Population(network, minTargetFlow)
    hillClimb.solveWithNaiveHillClimb(printGenerations=True, drawing=True, drawLabels=True)
    """

    # Set Hyperparameters
    pop.setPopulationHyperparams(populationSize=10, numGenerations=10,
                                 initializationDistribution="uniform", initializationParams=[0.0, 100.0])
    pop.setIndividualSelectionHyperparams(selectionMethod="tournament", tournamentSize=5)
    pop.setCrossoverHyperparams(crossoverMethod="onePoint", replacementStrategy="replaceParents")
    pop.setMutationHyperparams(mutationMethod="randomTotal", mutationRate=0.50)

    # Solve the Alpha-GA
    pop.evolvePopulation(printGenerations=True, drawing=True, drawLabels=True)

    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
    cplex.findSolution(printDetails=True)
    opt = cplex.writeSolution()
    optVis = SolutionVisualizer(opt)
    optVis.drawGraphWithLabels(leadingText="OPT_")

    # Solve with Naive Hill Climb
    # hillClimb = Population(network, minTargetFlow)
    # hillClimb.solveWithNaiveHillClimb(printGenerations=True, drawing=True, drawLabels=True)

    """
    # All hyperparameter setters
    pop.setPopulationHyperparams(populationSize=10, terminationMethod="setGenerations", numGenerations=1,
                                 stagnationPeriod=5, initializationDistribution="uniform",
                                 initializationParams=[0.0, 1.0])
    pop.setIndividualSelectionHyperparams(selectionMethod="tournament", tournamentSize=3)
    pop.setPathSelectionHyperparams(pathSelectionMethod="roulette", pathRankingOrder="most",
                                    pathRankingMethod="density", pathSelectionSize=2, pathTournamentSize=3)
    pop.setCrossoverHyperparams(crossoverMethod="pathBased", replacementStrategy="replaceParents",
                                crossoverRate=1.0, crossoverAttemptsPerGeneration=1)
    pop.setMutationHyperparams(mutationMethod="pathBasedNudge", mutationRate=0.25, nudgeParams=[0.0, 1.0])
    """
