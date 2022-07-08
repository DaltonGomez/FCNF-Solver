
from src.Experiments.GAvsMILP import GAvsMILP

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_gaVSmilp.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 run_gaVSmilp.py
"""

if __name__ == "__main__":
    # Input graph and experiment object w/ options
    inputGraph = "huge_2"
    gaVSmilp = GAvsMILP(inputGraph, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=True,
                              isOptimizedArcSelections=True, isSolvedWithMILP=True, isRace=True,
                              isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=True)

    # Alpha-GA population attribute & hyperparameters
    gaVSmilp.geneticPop.setPopulationHyperparams(populationSize=20,
                                             numGenerations=25,
                                             terminationMethod="setGenerations")
    gaVSmilp.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                 initializationDistribution="digital",
                                                 initializationParams=[2.0, 100000.0])
    gaVSmilp.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                      tournamentSize=4)
    gaVSmilp.geneticPop.setCrossoverHyperparams(crossoverMethod="onePoint",
                                            crossoverRate=1.0,
                                            crossoverAttemptsPerGeneration=1,
                                            replacementStrategy="replaceWeakestTwo")
    gaVSmilp.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                           mutationRate=0.05,
                                           perArcEdgeMutationRate=0.25)
    gaVSmilp.geneticPop.setDaemonHyperparams(isDaemonUsed=True, annealingConstant=0.5,
                                                  daemonStrategy="globalMean", daemonStrength=1)

    # Solve the graph
    gaVSmilp.solveGraph()
