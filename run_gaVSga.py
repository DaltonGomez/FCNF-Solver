
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
    gaVSga = GAvsGA(inputGraph, isPop1OneDimAlpha=True, isPop1ArcOptimized=True, isPop2OneDimAlpha=False,
                    isPop2ArcOptimized=True, isDrawing=True, isLabeling=True, isGraphing=True)

    # Alpha-GA population one hyperparameter setters
    gaVSga.geneticPopOne.setPopulationHyperparams(populationSize=10,
                                                numGenerations=25,
                                                terminationMethod="setGenerations")
    gaVSga.geneticPopOne.setInitializationHyperparams(initializationStrategy="perEdge",
                                                    initializationDistribution="digital",
                                                    initializationParams=[5.0, 100000.0])
    gaVSga.geneticPopOne.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                         tournamentSize=4)
    gaVSga.geneticPopOne.setCrossoverHyperparams(crossoverMethod="onePoint",
                                               crossoverRate=1.0,
                                               crossoverAttemptsPerGeneration=1,
                                               replacementStrategy="replaceWeakestTwo")
    gaVSga.geneticPopOne.setMutationHyperparams(mutationMethod="randomPerEdge",
                                              mutationRate=0.05,
                                              perArcEdgeMutationRate=0.25)
    gaVSga.geneticPopOne.setDaemonHyperparams(isDaemonUsed=True, annealingConstant=0.5,
                                                  daemonStrategy="globalMean", daemonStrength=1)

    # Alpha-GA population two hyperparameter setters
    gaVSga.geneticPopTwo.setPopulationHyperparams(populationSize=20,
                                                numGenerations=40,
                                                terminationMethod="setGenerations",
                                                stagnationPeriod=5)
    gaVSga.geneticPopTwo.setInitializationHyperparams(initializationStrategy="perArc",
                                                    initializationDistribution="digital",
                                                    initializationParams=[5.0, 100000.0])
    gaVSga.geneticPopTwo.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                         tournamentSize=4)
    gaVSga.geneticPopTwo.setCrossoverHyperparams(crossoverMethod="onePoint",
                                               crossoverRate=1.0,
                                               crossoverAttemptsPerGeneration=1,
                                               replacementStrategy="replaceWeakestTwo")
    gaVSga.geneticPopTwo.setMutationHyperparams(mutationMethod="randomPerEdge",
                                              mutationRate=0.05,
                                              perArcEdgeMutationRate=0.25)
    gaVSga.geneticPopTwo.setDaemonHyperparams(isDaemonUsed=True, annealingConstant=0.5,
                                              daemonStrategy="globalMean", daemonStrength=1)

    # Solve the graph
    gaVSga.solveGraph()
