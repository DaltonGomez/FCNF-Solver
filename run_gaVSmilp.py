
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
    inputGraph = "massive_2"
    gaVSmilp = GAvsMILP(inputGraph, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=True,
                        isOptimizedArcSelections=True, isSolvedWithMILP=False, isRace=True,
                        isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=True)

    # Alpha-GA population attribute & hyperparameters
    gaVSmilp.geneticPop.setPopulationHyperparams(populationSize=20,
                                                 numGenerations=20,
                                                 terminationMethod="setGenerations")
    gaVSmilp.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                     initializationDistribution="gaussian",
                                                     initializationParams=[1.0, 0.33])
    gaVSmilp.geneticPop.setIndividualSelectionHyperparams(selectionMethod="roulette",
                                                          tournamentSize=5)
    gaVSmilp.geneticPop.setCrossoverHyperparams(crossoverMethod="twoPoint",
                                                crossoverRate=1.0,
                                                crossoverAttemptsPerGeneration=1,
                                                replacementStrategy="appendOffspring")
    gaVSmilp.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                               mutationRate=0.20,
                                               mutationStrength=0.25)
    gaVSmilp.geneticPop.setDaemonHyperparams(isDaemonUsed=True, daemonAnnealingRate=0.5,
                                             daemonStrategy="globalMedian", daemonStrength=0.10)

    # Solve the graph
    gaVSmilp.solveGraph()
