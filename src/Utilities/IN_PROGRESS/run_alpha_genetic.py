from datetime import datetime

from src.AlphaGenetic.Population import Population
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_alpha_genetic.py
"""

if __name__ == "__main__":
    # Load Candidate Graph
    graphName = "small_2.p"
    graph = CandidateGraph()
    graph = graph.loadCandidateGraph(graphName)
    minTargetFlow = graph.totalPossibleDemand

    # Initialize an Alpha-GA Population
    pop = Population(graph, minTargetFlow, isOneDimAlphaTable=True, isOptimizedArcSelections=True)

    # Set Hyperparameters
    pop.setPopulationHyperparams(populationSize=20, numGenerations=20, initializationStrategy="perEdge",
                                 initializationDistribution="digital", initializationParams=[0.0, 100000.0])
    pop.setIndividualSelectionHyperparams(selectionMethod="tournament", tournamentSize=4)
    pop.setCrossoverHyperparams(crossoverMethod="twoPoint", crossoverRate=1.0, crossoverAttemptsPerGeneration=1,
                                replacementStrategy="replaceWeakestTwo")
    pop.setMutationHyperparams(mutationMethod="randomPerEdge", mutationRate=0.05, perArcEdgeMutationRate=0.25)

    # Timestamp the start of the GA evolution
    gaStartTime = datetime.now()
    print("\n\nSolving the " + graphName + " graph with a GA population of " + str(pop.populationSize) + " for " + str(pop.numGenerations) + " generations...")
    print("GA Start: " + str(gaStartTime) + "\n")

    # Solve the Alpha-GA
    gaSolution = pop.evolvePopulation(printGenerations=True, drawing=True, drawLabels=False)
    print("\nBest solution found = " + str(gaSolution.trueCost))
    solVis = SolutionVisualizer(gaSolution)
    solVis.drawUnlabeledSolution(leadingText="GA-BEST_")

    # Timestamp the finish of the GA evolution/start of the optimal MILP solver
    gaFinishOptStart = datetime.now()
    print("\nGA Finish/OPT Start: " + str(gaFinishOptStart))
    gaRuntime = gaFinishOptStart - gaStartTime
    print("\nGA Runtime (in seconds): " + str(gaRuntime.seconds))

    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(graph, minTargetFlow, isOneArcPerEdge=False, logOutput=False)
    cplex.setTimeLimit(gaRuntime.seconds)
    cplex.findSolution(printDetails=False)
    opt = cplex.writeSolution()
    optVis = SolutionVisualizer(opt)
    optVis.drawUnlabeledSolution(leadingText="OPT_")

    # Timestamp the finish of the optimal MILP solver
    optFinish = datetime.now()
    print("\nOPT Finish: " + str(optFinish))
    optRuntime = optFinish - gaFinishOptStart
    print("OPT Runtime (in seconds): " + str(optRuntime.seconds))

    print("\n\nProgram complete! Terminating...\n")
