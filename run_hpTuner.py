
from src.Experiments.HyperparamTuner import HyperparamTuner

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_hpTuner.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 run_hpTuner.py
"""


if __name__ == "__main__":
    # Input graph and tuner object w/ options
    inputGraphs = ["huge_0",
                   "huge_1",
                   "huge_2",
                   "huge_3",
                   "huge_4",
                   "huge_5",
                   "huge_6",
                   "huge_7",
                   "huge_8",
                   "huge_9",
                   "massive_0",
                   "massive_1",
                   "massive_2",
                   "massive_3",
                   "massive_4",
                   "massive_5",
                   "massive_6",
                   "massive_7",
                   "massive_8",
                   "massive_9"
                   ]
    runsPerGraph = 3
    hpTuner = HyperparamTuner(inputGraphs, runsPerGraph, isDaemonUsed=True,
                              tuneOneDimAlpha=True, tuneManyDimAlpha=False,
                              tuneOptimizedArcs=True, tuneNonOptimizedArcs=False)

    # Update the HP tuning search space
    hpTuner.hpSpace = {
                "populationSize": [20],
                "numGenerations": [30],
                "initializationStrategy": ["perEdge"],
                "initializationDistribution": ["gaussian"],
                "initializationParams": [
                                            [500.0, 100.0],
                                        ],
                "selectionMethod": ["tournament"],
                "tournamentSize": [5],
                "crossoverMethod": ["onePoint"],
                "crossoverRate": [1.0],
                "crossoverAttemptsPerGeneration": [2],
                "replacementStrategy": ["replaceWeakestTwo"],
                "mutationMethod": ["randomPerEdge"],
                "mutationRate": [0.05, 0.10, 0.25],
                "perArcEdgeMutationRate": [0.25, 0.50],
                "isDaemonUsed": [True],
                "annealingConstant": [0.10, 0.25, 0.5],
                "daemonStrategy": ["globalMedian", "personalMedian"],
                "daemonStrength": [0.10, 0.50]
                }

    # Solve the graph
    # NOTE - The grid search is not comprehensive of all the hyperparameters that could be tuned
    hpTuner.runTuningExperiment()


"""
# COMPLETE HP SEARCH-SPACE DICTIONARY:
self.hpSpace: Dict[str, List] = {
                        "populationSize": [10, 25, 50, 100],
                        "numGenerations": [10, 25, 50, 100],
                        "terminationMethod": ["setGenerations", "stagnationPeriod"],
                        "stagnationPeriod": [5],
                        "initializationStrategy": ["perEdge", "perArc", "reciprocalCap"],
                        "initializationDistribution": ["uniform", "gaussian", "digital"],
                        "initializationParams": [
                                                    [0, 100000],
                                                    [1, 100000],
                                                    [10, 100000]
                                                ],
                        "selectionMethod": ["tournament", "roulette", "random"],
                        "tournamentSize": [3, 5, 8],
                        "crossoverMethod": ["onePoint", "twoPoint"],
                        "crossoverRate": [0.25, 0.50, 0.75, 1.0],
                        "crossoverAttemptsPerGeneration": [1, 2, 3, 4],
                        "replacementStrategy": ["replaceWeakestTwo", "replaceParents"],
                        "mutationMethod": ["randomSingleArc", "randomSingleEdge", "randomPerArc", "randomPerEdge", "randomTotal"],
                        "mutationRate": [0.01, 0.05, 0.10, 0.25, 0.50],
                        "perArcEdgeMutationRate": [0.01, 0.05, 0.10, 0.25, 0.50],
                        "isDaemonUsed": [True, False],
                        "annealingConstant": [0.25, 0.5, 1, 2],
                        "daemonStrategy": ["globalBinary", "globalMean", "globalMedian", "personalMean", "personalMedian"],
                        "daemonStrength": [0.5, 1, 2]
                        }
"""

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
