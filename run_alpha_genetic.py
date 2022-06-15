from src.AlphaGenetic.Population import Population
from src.FlowNetwork.CandidateGraph import CandidateGraph
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_alpha_genetic.py
"""

if __name__ == "__main__":
    # Load FlowNetwork
    graph = CandidateGraph()
    graph = graph.loadCandidateGraph("medium_2.p")
    minTargetFlow = graph.totalPossibleDemand

    # Initialize an Alpha-GA Population
    pop = Population(graph, minTargetFlow)

    # Set Hyperparameters
    pop.setPopulationHyperparams(populationSize=100, numGenerations=100, initializationStrategy="perEdge",
                                 initializationDistribution="digital", initializationParams=[0.0, 200000.0])
    pop.setIndividualSelectionHyperparams(selectionMethod="tournament", tournamentSize=5)
    pop.setCrossoverHyperparams(crossoverMethod="onePoint", replacementStrategy="replaceWeakestTwo", crossoverRate=1.0,
                                crossoverAttemptsPerGeneration=3)
    pop.setMutationHyperparams(mutationMethod="randomPerEdge", mutationRate=0.05, perArcEdgeMutationRate=0.20)

    # Solve the Alpha-GA
    solutionTuple = pop.evolvePopulation(printGenerations=True, drawing=True, drawLabels=True)
    print("Best solution found = " + str(solutionTuple[0]))
    solVis = SolutionVisualizer(solutionTuple[1])
    solVis.drawLabeledSolution(leadingText="GA_best_")

    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(graph, minTargetFlow, isOneArcPerEdge=False)
    cplex.findSolution(printDetails=True)
    opt = cplex.writeSolution()
    optVis = SolutionVisualizer(opt)
    optVis.drawLabeledSolution(leadingText="OPT_")

    """
    # TODO - Update as these are out of date
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
    # Solve with Naive Hill Climb
    hillClimb = Population(graph, minTargetFlow)
    hillClimb.solveWithNaiveHillClimb(printGenerations=True, drawing=True, drawLabels=True)
    """
