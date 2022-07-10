
from src.Experiments.MultiGAvsMILP import MultiGAvsMILP

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_multi_gaVSmilp.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 run_multi_gaVSmilp.py
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
                   "massive_UB_9"
                   ]
    # Option to reverse input graphs list (i.e. run large graphs first)
    inputGraphs.reverse()
    # Number of runs per graph
    runsPerGraph = 3

    # Experiment object and hyperparameter settings
    multiGAvsMILP = MultiGAvsMILP(inputGraphs, runsPerGraph, isSolvedWithGeneticAlg=True,
                                    isOneDimAlphaTable=True, isOptimizedArcSelections=True,
                                    isSolvedWithMILP=True, isRace=True)
    multiGAvsMILP.populationSize = 10
    multiGAvsMILP.numGenerations = 10
    multiGAvsMILP.terminationMethod = "setGenerations"
    multiGAvsMILP.initializationStrategy = "perEdge"
    multiGAvsMILP.initializationDistribution = "digital"
    multiGAvsMILP.initializationParams = [5.0, 100000.0]
    multiGAvsMILP.selectionMethod = "tournament"
    multiGAvsMILP.tournamentSize = 4
    multiGAvsMILP.crossoverMethod = "onePoint"
    multiGAvsMILP.crossoverRate = 1.0
    multiGAvsMILP.crossoverAttemptsPerGeneration = 1
    multiGAvsMILP.replacementStrategy = "replaceWeakestTwo"
    multiGAvsMILP.mutationMethod = "randomPerEdge"
    multiGAvsMILP.mutationRate = 0.05
    multiGAvsMILP.perArcEdgeMutationRate = 0.25
    multiGAvsMILP.isDaemonUsed = True
    multiGAvsMILP.annealingConstant = 0.5
    multiGAvsMILP.daemonStrategy = "globalMean"
    multiGAvsMILP.daemonStrength = 1

    # Execute all runs
    multiGAvsMILP.runSolversOnAllGraphs()

"""
# COMPLETE INPUT GRAPHS LIST
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
               "massive_UB_9"
               ]
"""