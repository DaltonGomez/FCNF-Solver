from Network.SolutionVisualizer import SolutionVisualizer
from Solvers.MILPsolverCPLEX import MILPsolverCPLEX
from src.AlphaGenetic.Population import Population
from src.Network.FlowNetwork import FlowNetwork

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_alpha_genetic.py
"""

if __name__ == "__main__":
    # Load Network
    network = FlowNetwork()
    network = network.loadNetwork("cluster_test_1.p")
    minTargetFlow = network.totalPossibleDemand

    # Initialize an Alpha-GA Population
    pop = Population(network, minTargetFlow)

    # Set Hyperparameters
    pop.setPopulationHyperparams(populationSize=100, numGenerations=10,
                                 initializationDistribution="uniform", initializationParams=[0.0, 100.0])
    pop.setIndividualSelectionHyperparams(selectionMethod="tournament", tournamentSize=5)
    pop.setCrossoverHyperparams(crossoverMethod="onePoint", replacementStrategy="replaceWeakestTwo")
    pop.setMutationHyperparams(mutationMethod="randomPerArc", mutationRate=0.10, perArcEdgeMutationRate=0.25)

    # Solve the Alpha-GA
    solutionTuple = pop.evolvePopulation(printGenerations=True, drawing=True, drawLabels=True)

    print("Best solution found = " + str(solutionTuple[0]))
    solVis = SolutionVisualizer(solutionTuple[1])
    solVis.drawGraphWithLabels(leadingText="GA_best_")

    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
    cplex.findSolution(printDetails=True)
    opt = cplex.writeSolution()
    optVis = SolutionVisualizer(opt)
    optVis.drawGraphWithLabels(leadingText="OPT_")

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

    """
    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
    cplex.findSolution(printDetails=True)
    opt = cplex.writeSolution()
    optVis = SolutionVisualizer(opt)
    optVis.drawGraphWithLabels(leadingText="OPT_")
    
    # Solve with Naive Hill Climb
    hillClimb = Population(network, minTargetFlow)
    hillClimb.solveWithNaiveHillClimb(printGenerations=True, drawing=True, drawLabels=True)
    """
