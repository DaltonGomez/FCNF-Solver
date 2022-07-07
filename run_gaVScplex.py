
from src.Experiments.GAvsCPLEX import GAvsCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_gaVScplex.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 run_gaVScplex.py
"""

if __name__ == "__main__":
    # Input graph and experiment object w/ options
    inputGraph = "huge_2"
    gaVScplex = GAvsCPLEX(inputGraph, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=True,
                              isOptimizedArcSelections=True, isSolvedWithCPLEX=True, isRace=True,
                              isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=True)

    # Alpha-GA population attribute & hyperparameters
    gaVScplex.geneticPop.setPopulationHyperparams(populationSize=10,
                                             numGenerations=10,
                                             terminationMethod="setGenerations",
                                             stagnationPeriod=5)
    gaVScplex.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                 initializationDistribution="digital",
                                                 initializationParams=[0.0, 100000.0])
    gaVScplex.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                      tournamentSize=4)
    gaVScplex.geneticPop.setCrossoverHyperparams(crossoverMethod="onePoint",
                                            crossoverRate=1.0,
                                            crossoverAttemptsPerGeneration=1,
                                            replacementStrategy="replaceWeakestTwo")
    gaVScplex.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                           mutationRate=0.05,
                                           perArcEdgeMutationRate=0.25)

    # Solve the graph
    gaVScplex.solveGraph()
