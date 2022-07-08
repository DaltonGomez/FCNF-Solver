
from src.Experiments.GAvsMILP import GAvsMILP

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 sandbox.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 sandbox.py
"""

if __name__ == "__main__":
    # Input graph and experiment object w/ options
    inputGraph = "huge_UB_6"
    sandboxSolver = GAvsMILP(inputGraph, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=True,
                              isOptimizedArcSelections=True, isSolvedWithMILP=True, isRace=True,
                              isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=True)

    # Alpha-GA population attribute & hyperparameters
    sandboxSolver.geneticPop.setPopulationHyperparams(populationSize=25,
                                             numGenerations=50,
                                             terminationMethod="setGenerations")
    sandboxSolver.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                 initializationDistribution="digital",
                                                 initializationParams=[5.0, 1000000.0])
    sandboxSolver.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                      tournamentSize=3)
    sandboxSolver.geneticPop.setCrossoverHyperparams(crossoverMethod="onePoint",
                                            crossoverRate=1.0,
                                            crossoverAttemptsPerGeneration=1,
                                            replacementStrategy="replaceWeakestTwo")
    sandboxSolver.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                           mutationRate=0.10,
                                           perArcEdgeMutationRate=0.25)
    sandboxSolver.geneticPop.setDaemonHyperparams(isDaemonUsed=True, annealingConstant=0.5,
                                                  daemonStrategy="globalMean", daemonStrength=1)
    # Solve the graph
    sandboxSolver.solveGraph()
