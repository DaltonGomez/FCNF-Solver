
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
    inputGraph = "massive_UB_0"
    sandboxSolver = GAvsMILP(inputGraph, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=False,
                              isOptimizedArcSelections=False, isSolvedWithMILP=True, isRace=True,
                              isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=True)

    # Alpha-GA population attribute & hyperparameters
    sandboxSolver.geneticPop.setPopulationHyperparams(populationSize=10,
                                             numGenerations=20,
                                             terminationMethod="setGenerations")
    sandboxSolver.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                 initializationDistribution="gaussian",
                                                 initializationParams=[500.0, 100.0])
    sandboxSolver.geneticPop.setIndividualSelectionHyperparams(selectionMethod="random",
                                                      tournamentSize=3)
    sandboxSolver.geneticPop.setCrossoverHyperparams(crossoverMethod="twoPoint",
                                            crossoverRate=1.0,
                                            crossoverAttemptsPerGeneration=2,
                                            replacementStrategy="replaceWeakestTwo")
    sandboxSolver.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                           mutationRate=0.20,
                                           perArcEdgeMutationRate=0.25)
    sandboxSolver.geneticPop.setDaemonHyperparams(isDaemonUsed=False, annealingConstant=0.10,
                                                  daemonStrategy="globalMean", daemonStrength=2)
    # Solve the graph
    sandboxSolver.solveGraph()
