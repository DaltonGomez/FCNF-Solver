from datetime import datetime

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
    # Load Candidate Graph
    graphName = "test_8.p"
    graph = CandidateGraph()
    graph = graph.loadCandidateGraph(graphName)
    minTargetFlow = graph.totalPossibleDemand

    # Initialize an Alpha-GA Population
    pop = Population(graph, minTargetFlow)

    # Set Hyperparameters
    pop.setPopulationHyperparams(populationSize=10, numGenerations=3, initializationStrategy="perEdge",
                                 initializationDistribution="digital", initializationParams=[0.0, 200000.0])
    pop.setIndividualSelectionHyperparams(selectionMethod="tournament", tournamentSize=3)
    pop.setCrossoverHyperparams(crossoverMethod="onePoint", crossoverRate=1.0, crossoverAttemptsPerGeneration=3,
                                replacementStrategy="replaceWeakestTwo")
    pop.setMutationHyperparams(mutationMethod="randomPerEdge", mutationRate=0.05, perArcEdgeMutationRate=0.20)

    # Timestamp the start of the GA evolution
    gaStartTime = datetime.now()
    print("\n\nSolving the " + graphName + " graph with a GA population of " + str(pop.populationSize) + " for " + str(pop.numGenerations) + " generations...")
    print("GA Start: " + str(gaStartTime) + "\n")

    # Solve the Alpha-GA
    solutionTuple = pop.evolvePopulation(printGenerations=True, drawing=True, drawLabels=True)
    print("\nBest solution found = " + str(solutionTuple[0]))
    solVis = SolutionVisualizer(solutionTuple[1])
    solVis.drawLabeledSolution(leadingText="GA_best_")

    # Timestamp the finish of the GA evolution/start of the optimal MILP solver
    gaFinishOptStart = datetime.now()
    print("\nGA Finish/OPT Start: " + str(gaFinishOptStart))
    gaRuntime = gaFinishOptStart - gaStartTime
    print("GA Runtime in Minutes: " + str(gaRuntime.seconds / 60))

    # Solve Optimally with CPLEX
    cplex = MILPsolverCPLEX(graph, minTargetFlow, isOneArcPerEdge=False)
    cplex.findSolution(printDetails=True)
    opt = cplex.writeSolution()
    optVis = SolutionVisualizer(opt)
    optVis.drawLabeledSolution(leadingText="OPT_")

    # Timestamp the finish of the optimal MILP solver
    optFinish = datetime.now()
    print("\nOPT Finish: " + str(optFinish))
    optRuntime = optFinish - gaFinishOptStart
    print("OPT Runtime in Minutes: " + str(optRuntime.seconds / 60))

    print("\n\nProgram complete! Terminating...")
