
from src.Experiments.GAvsNaive import GAvsNaive

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
    inputGraph = "massive_2"
    gaVSmilp = GAvsNaive(inputGraph, isSolvedWithGeneticAlg=True, isSolvedWithNaive=True, isOneDimAlphaTable=True,
                         isOptimizedArcSelections=True, isDrawing=False, isLabeling=True, isGraphing=True)

    # Alpha-GA population attribute & hyperparameters
    gaVSmilp.geneticPop.setPopulationHyperparams(populationSize=20,
                                                 numGenerations=200,
                                                 terminationMethod="setGenerations")
    gaVSmilp.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                     initializationDistribution="gaussian",
                                                     initializationParams=[500.0, 100.0])
    gaVSmilp.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                          tournamentSize=5)
    gaVSmilp.geneticPop.setCrossoverHyperparams(crossoverMethod="twoPoint",
                                                crossoverRate=1.0,
                                                crossoverAttemptsPerGeneration=1,
                                                replacementStrategy="replaceWeakestTwo")
    gaVSmilp.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                               mutationRate=0.20,
                                               mutationStrength=0.25)
    gaVSmilp.geneticPop.setDaemonHyperparams(isDaemonUsed=True, daemonAnnealingRate=0.10,
                                             daemonStrategy="globalMedian", daemonStrength=0.10)

    # Naive-HC population attribute & hyperparameters
    gaVSmilp.naivePop.setPopulationHyperparams(populationSize=20,
                                                 numGenerations=50,
                                                 terminationMethod="setGenerations")

    # Solve the graph
    gaVSmilp.solveGraph()
