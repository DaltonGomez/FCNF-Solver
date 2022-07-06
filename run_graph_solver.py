
from src.Experiments.GraphSolver import GraphSolver

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_graph_solver.py
"""

if __name__ == "__main__":
    # Input graph and solver object w/ options
    inputGraph = "huge_2.p"
    graphSolver = GraphSolver(inputGraph, isSolvedWithGeneticAlg=True, isSolvedWithCPLEX=True, isRace=True,
                              isDrawing=False, isLabeling=True, isGraphing=True, isOutputtingCPLEX=False)

    # Options to adjust the GA hyperparameters
    graphSolver.geneticPop.setPopulationHyperparams(populationSize=5,
                                             numGenerations=5,
                                             initializationStrategy="perEdge",
                                             initializationDistribution="digital",
                                             initializationParams=[0.0, 100000.0])
    graphSolver.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                      tournamentSize=4)
    graphSolver.geneticPop.setCrossoverHyperparams(crossoverMethod="onePoint",
                                            crossoverRate=1.0,
                                            crossoverAttemptsPerGeneration=1,
                                            replacementStrategy="replaceWeakestTwo")
    graphSolver.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                           mutationRate=0.05,
                                           perArcEdgeMutationRate=0.25)

    # Solve the graph
    graphSolver.solveGraph()
