
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
    inputGraph = "test_0"
    gaVSmilp = GAvsMILP(inputGraph, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=True,
                        isOptimizedArcSelections=True, isSolvedWithMILP=True, isRace=True,
                        isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=True)

    # Alpha-GA population attribute & hyperparameters
    gaVSmilp.geneticPop.setPopulationHyperparams(populationSize=25,
                                                 numGenerations=25,
                                                 terminationMethod="setGenerations")
    gaVSmilp.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                     initializationDistribution="gaussian",
                                                     initializationParams=[500.0, 100.0])
    gaVSmilp.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                          tournamentSize=3)
    gaVSmilp.geneticPop.setCrossoverHyperparams(crossoverMethod="twoPoint",
                                                crossoverRate=1.0,
                                                crossoverAttemptsPerGeneration=2,
                                                replacementStrategy="replaceWeakestTwo")
    gaVSmilp.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                               mutationRate=0.10,
                                               perArcEdgeMutationRate=0.25)
    gaVSmilp.geneticPop.setDaemonHyperparams(isDaemonUsed=True, annealingConstant=0.10,
                                             daemonStrategy="globalMedian", daemonStrength=0.10)

    # Solve the graph
    gaVSmilp.solveGraph()
