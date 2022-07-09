
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
    inputGraph = "huge_2"
    gaVSga = GAvsGA(inputGraph, isPop1OneDimAlpha=True, isPop1ArcOptimized=True, isPop1Penalized=False,
                    isPop2OneDimAlpha=True, isPop2ArcOptimized=True, isPop2Penalized=False,
                    isDrawing=True, isLabeling=True, isGraphing=True)

    # Alpha-GA population one hyperparameter setters
    gaVSga.geneticPopOne.setPopulationHyperparams(populationSize=1,
                                                numGenerations=20,
                                                terminationMethod="setGenerations")
    gaVSga.geneticPopOne.setInitializationHyperparams(initializationStrategy="perEdge",
                                                    initializationDistribution="gaussian",
                                                    initializationParams=[500.0, 100.0])
    gaVSga.geneticPopOne.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                         tournamentSize=3)
    gaVSga.geneticPopOne.setCrossoverHyperparams(crossoverMethod="onePoint",
                                               crossoverRate=1.0,
                                               crossoverAttemptsPerGeneration=0,
                                               replacementStrategy="replaceWeakestTwo")
    gaVSga.geneticPopOne.setMutationHyperparams(mutationMethod="randomPerEdge",
                                              mutationRate=0.00,
                                              perArcEdgeMutationRate=0.25)
    gaVSga.geneticPopOne.setDaemonHyperparams(isDaemonUsed=True, annealingConstant=4,
                                                  daemonStrategy="personalMedian", daemonStrength=0.10)

    # Alpha-GA population two hyperparameter setters
    gaVSga.geneticPopTwo.setPopulationHyperparams(populationSize=1,
                                                numGenerations=20,
                                                terminationMethod="setGenerations")
    gaVSga.geneticPopTwo.setInitializationHyperparams(initializationStrategy="perEdge",
                                                      initializationDistribution="gaussian",
                                                      initializationParams=[500.0, 100.0])
    gaVSga.geneticPopTwo.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                         tournamentSize=3)
    gaVSga.geneticPopTwo.setCrossoverHyperparams(crossoverMethod="onePoint",
                                               crossoverRate=1.0,
                                               crossoverAttemptsPerGeneration=0,
                                               replacementStrategy="replaceWeakestTwo")
    gaVSga.geneticPopTwo.setMutationHyperparams(mutationMethod="randomPerEdge",
                                              mutationRate=0.00,
                                              perArcEdgeMutationRate=0.25)
    gaVSga.geneticPopTwo.setDaemonHyperparams(isDaemonUsed=True, annealingConstant=4,
                                              daemonStrategy="personalMedian", daemonStrength=0.10)

    # Solve the graph
    gaVSga.solveGraph()
