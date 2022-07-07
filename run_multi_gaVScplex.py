
from src.Experiments.MultiGAvsCPLEX import MultiGAvsCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_multi_gaVScplex.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 run_multi_gaVScplex.py
"""

if __name__ == "__main__":
    # Input graphs
    inputGraphs = ["test_0",
                   "test_1",
                   "test_2",
                   "test_3",
                   "test_4",
                   "test_5",
                   "test_6",
                   "test_7",
                   "test_8",
                   "test_9",
                   "test_UB_0",
                   "test_UB_1",
                   "test_UB_2",
                   "test_UB_3",
                   "test_UB_4",
                   "test_UB_5",
                   "test_UB_6",
                   "test_UB_7",
                   "test_UB_8",
                   "test_UB_9",
                   "small_0",
                   "small_1",
                   "small_2",
                   "small_3",
                   "small_4",
                   "small_5",
                   "small_6",
                   "small_7",
                   "small_8",
                   "small_9",
                   "small_UB_0",
                   "small_UB_1",
                   "small_UB_2",
                   "small_UB_3",
                   "small_UB_4",
                   "small_UB_5",
                   "small_UB_6",
                   "small_UB_7",
                   "small_UB_8",
                   "small_UB_9",
                   "medium_0",
                   "medium_1",
                   "medium_2",
                   "medium_3",
                   "medium_4",
                   "medium_5",
                   "medium_6",
                   "medium_7",
                   "medium_8",
                   "medium_9",
                   "medium_UB_0",
                   "medium_UB_1",
                   "medium_UB_2",
                   "medium_UB_3",
                   "medium_UB_4",
                   "medium_UB_5",
                   "medium_UB_6",
                   "medium_UB_7",
                   "medium_UB_8",
                   "medium_UB_9",
                   "large_0",
                   "large_1",
                   "large_2",
                   "large_3",
                   "large_4",
                   "large_5",
                   "large_6",
                   "large_7",
                   "large_8",
                   "large_9",
                   "large_UB_0",
                   "large_UB_1",
                   "large_UB_2",
                   "large_UB_3",
                   "large_UB_4",
                   "large_UB_5",
                   "large_UB_6",
                   "large_UB_7",
                   "large_UB_8",
                   "large_UB_9",
                   "huge_0",
                   "huge_1",
                   "huge_2",
                   "huge_3",
                   "huge_4",
                   "huge_5",
                   "huge_6",
                   "huge_7",
                   "huge_8",
                   "huge_9",
                   "huge_UB_0",
                   "huge_UB_1",
                   "huge_UB_2",
                   "huge_UB_3",
                   "huge_UB_4",
                   "huge_UB_5",
                   "huge_UB_6",
                   "huge_UB_7",
                   "huge_UB_8",
                   "huge_UB_9",
                   "massive_0",
                   "massive_1",
                   "massive_2",
                   "massive_3",
                   "massive_4",
                   "massive_5",
                   "massive_6",
                   "massive_7",
                   "massive_8",
                   "massive_9",
                   "massive_UB_0",
                   "massive_UB_1",
                   "massive_UB_2",
                   "massive_UB_3",
                   "massive_UB_4",
                   "massive_UB_5",
                   "massive_UB_6",
                   "massive_UB_7",
                   "massive_UB_8",
                   "massive_UB_9",
                   ]
    # Number of runs per graph
    runsPerGraph = 3

    # Experiment object and hyperparameter settings
    multiGAvsCPLEX = MultiGAvsCPLEX(inputGraphs, runsPerGraph, isSolvedWithGeneticAlg=True,
                                    isOneDimAlphaTable=True, isOptimizedArcSelections=True,
                                    isSolvedWithCPLEX=True, isRace=True)
    multiGAvsCPLEX.populationSize = 20
    multiGAvsCPLEX.numGenerations = 20
    multiGAvsCPLEX.terminationMethod = "setGenerations"
    multiGAvsCPLEX.stagnationPeriod = 5
    multiGAvsCPLEX.initializationStrategy = "perEdge"
    multiGAvsCPLEX.initializationDistribution = "digital"
    multiGAvsCPLEX.initializationParams = [0.0, 100000.0]
    multiGAvsCPLEX.selectionMethod = "tournament"
    multiGAvsCPLEX.tournamentSize = 4
    multiGAvsCPLEX.crossoverMethod = "onePoint"
    multiGAvsCPLEX.crossoverRate = 1.0
    multiGAvsCPLEX.crossoverAttemptsPerGeneration = 1
    multiGAvsCPLEX.replacementStrategy = "replaceWeakestTwo"
    multiGAvsCPLEX.mutationMethod = "randomPerEdge"
    multiGAvsCPLEX.mutationRate = 0.05
    multiGAvsCPLEX.perArcEdgeMutationRate = 0.25

    # Execute all runs
    multiGAvsCPLEX.runSolversOnAllGraphs()
