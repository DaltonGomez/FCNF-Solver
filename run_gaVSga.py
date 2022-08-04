
from src.Experiments.GAvsGA import GAvsGA

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_gaVSga.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 run_gaVSga.py
"""

if __name__ == "__main__":
    # Input graph and experiment object w/ options
    inputGraph = "massive_2"
    gaVSga = GAvsGA(inputGraph, isPop1OneDimAlpha=True, isPop1ArcOptimized=True,
                    isPop2OneDimAlpha=True, isPop2ArcOptimized=True,
                    isDrawing=True, isLabeling=True, isGraphing=True)

    # Alpha-GA population one hyperparameter setters
    gaVSga.geneticPopOne.setPopulationHyperparams(populationSize=20,
                                                  numGenerations=40,
                                                  terminationMethod="setGenerations")
    gaVSga.geneticPopOne.setInitializationHyperparams(initializationStrategy="perEdge",
                                                    initializationDistribution="gaussian",
                                                    initializationParams=[1.0, 0.25])
    gaVSga.geneticPopOne.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                           tournamentSize=5)
    gaVSga.geneticPopOne.setCrossoverHyperparams(crossoverMethod="twoPoint",
                                                 crossoverRate=1.0,
                                                 crossoverAttemptsPerGeneration=1,
                                                 replacementStrategy="replaceWeakestTwo")
    gaVSga.geneticPopOne.setMutationHyperparams(mutationMethod="randomPerEdge",
                                                mutationRate=0.10,
                                                mutationStrength=0.25)
    gaVSga.geneticPopOne.setDaemonHyperparams(isDaemonUsed=True, daemonAnnealingRate=0.50,
                                              daemonStrategy="globalMedian", daemonStrength=0.10)

    # Alpha-GA population two hyperparameter setters
    gaVSga.geneticPopTwo.setPopulationHyperparams(populationSize=20,
                                                  numGenerations=40,
                                                  terminationMethod="setGenerations")
    gaVSga.geneticPopTwo.setInitializationHyperparams(initializationStrategy="perEdge",
                                                      initializationDistribution="gaussian",
                                                      initializationParams=[50000.0, 10000.0])
    gaVSga.geneticPopTwo.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                           tournamentSize=5)
    gaVSga.geneticPopTwo.setCrossoverHyperparams(crossoverMethod="twoPoint",
                                                 crossoverRate=1.0,
                                                 crossoverAttemptsPerGeneration=1,
                                                 replacementStrategy="replaceWeakestTwo")
    gaVSga.geneticPopTwo.setMutationHyperparams(mutationMethod="randomPerEdge",
                                                mutationRate=0.10,
                                                mutationStrength=0.25)
    gaVSga.geneticPopTwo.setDaemonHyperparams(isDaemonUsed=True, daemonAnnealingRate=0.50,
                                              daemonStrategy="globalMedian", daemonStrength=0.10)

    # Solve the graph
    gaVSga.solveGraph()
