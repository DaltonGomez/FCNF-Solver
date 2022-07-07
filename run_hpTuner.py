
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
    inputGraphs = [
                "large_UB_0",
                "large_UB_1",
                "large_UB_2",
                "large_UB_3",
                "large_UB_4",
                "large_UB_5",
                "large_UB_6",
                "large_UB_7",
                "large_UB_8",
                "large_UB_9"]
    runsPerGraph = 3
    hpTuner = HyperparamTuner(inputGraphs, runsPerGraph)

    # Update the HP tuning search space
    hpTuner.hpSpace = {
                "isOneDimAlphaTable": [True],
                "isOptimizedArcSelections": [True],
                "populationSize": [10, 25, 50, 100],
                "numGenerations": [10, 25, 50, 100],
                "initializationStrategy": ["perEdge"],
                "initializationDistribution": ["digital"],
                "initializationParams": [
                                            [0, 100000],
                                            [1, 100000],
                                            [10, 100000]
                                        ],
                "selectionMethod": ["tournament", "roulette", "random"],
                "tournamentSize": [3, 5, 8],
                "crossoverMethod": ["onePoint", "twoPoint"],
                "crossoverRate": [0.70, 1.0],
                "crossoverAttemptsPerGeneration": [1, 3],
                "replacementStrategy": ["replaceWeakestTwo", "replaceParents"],
                "mutationMethod": ["randomPerEdge", "randomTotal"],
                "mutationRate": [0.01, 0.05, 0.10, 0.25, 0.50],
                "perArcEdgeMutationRate": [0.01, 0.05, 0.10, 0.25, 0.50]
                }

    # Solve the graph
    # NOTE - The grid search is not comprehensive of all the hyperparameters that could be tuned
    hpTuner.conductGridSearch()


"""
# COMPLETE HP SEARCH-SPACE DICTIONARY:
self.hpSpace: Dict[str, List] = {
                        "populationSize": [10, 25, 50, 100],
                        "numGenerations": [10, 25, 50, 100],
                        "terminationMethod": ["setGenerations", "stagnationPeriod"],
                        "stagnationPeriod": [5],
                        "isOneDimAlphaTable": [True, False],
                        "isOptimizedArcSelections": [True, False],
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
                        "perArcEdgeMutationRate": [0.01, 0.05, 0.10, 0.25, 0.50]
                        }
"""
