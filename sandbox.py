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
    inputGraph = "small_UB_0"
    sandboxSolver = GAvsMILP(inputGraph, isSolvedWithGeneticAlg=False, isOneDimAlphaTable=True,
                              isOptimizedArcSelections=True, isSolvedWithMILP=True, isRace=False,
                              isDrawing=True, isLabeling=False, isGraphing=True, isOutputtingCPLEX=True)

    # Alpha-GA population attribute & hyperparameters
    sandboxSolver.geneticPop.setPopulationHyperparams(populationSize=20,
                                             numGenerations=40,
                                             terminationMethod="setGenerations")
    sandboxSolver.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                 initializationDistribution="gaussian",
                                                 initializationParams=[500.0, 100.0])
    sandboxSolver.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                      tournamentSize=5)
    sandboxSolver.geneticPop.setCrossoverHyperparams(crossoverMethod="twoPoint",
                                            crossoverRate=1.0,
                                            crossoverAttemptsPerGeneration=1,
                                            replacementStrategy="replaceWeakestTwo")
    sandboxSolver.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                           mutationRate=0.20,
                                           mutationStrength=0.25)
    sandboxSolver.geneticPop.setDaemonHyperparams(isDaemonUsed=True, daemonAnnealingRate=0.10,
                                                  daemonStrategy="globalMedian", daemonStrength=0.10)
    # Solve the graph
    sandboxSolver.solveGraph()

    """
    # DEMOS WHY THE POST-PROCESSING ARC OPTIMIZATION MAKES THE GA SO GOOD AS IT EVEN MAKES A SUBSTANTIAL DIFFERENCE ON THE MILP
    rawMILPArcFlows = sandboxSolver.milpCplexSolver.model.solution.get_value_dict(sandboxSolver.milpCplexSolver.arcFlowVars)
    optMILPArcFlows = sandboxSolver.milpCplexSolver.optimizeArcSelection(sandboxSolver.milpCplexSolver.model.solution.get_value_dict(sandboxSolver.milpCplexSolver.arcFlowVars))

    for arcCapTup in rawMILPArcFlows.keys():
        if rawMILPArcFlows[arcCapTup] > 0.0:
            print("RAW")
            print(arcCapTup)
            print(rawMILPArcFlows[arcCapTup])
        if optMILPArcFlows[arcCapTup] > 0.0:
            print("OPT")
            print(arcCapTup)
            print(optMILPArcFlows[arcCapTup])
    """