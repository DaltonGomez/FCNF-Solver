
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
    # Input graphs and runs per graph
    inputGraphs = ["huge_0",
                   "huge_1",
                   "huge_2",
                   ]
    runsPerGraph = 1

    # Experiment object and hyperparameter settings
    multiGAvsCPLEX = MultiGAvsCPLEX(inputGraphs, runsPerGraph, isSolvedWithGeneticAlg=True,
                                    isOneDimAlphaTable=True, isOptimizedArcSelections=True,
                                    isSolvedWithCPLEX=True, isRace=True)
    multiGAvsCPLEX.populationSize = 10
    multiGAvsCPLEX.numGenerations = 10
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
