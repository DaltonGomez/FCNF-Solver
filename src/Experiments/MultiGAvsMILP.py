import csv
import os
from datetime import datetime
from typing import List

from src.Experiments.GAvsMILP import GAvsMILP


class MultiGAvsMILP:
    """Class that solves multiple graphs using the alpha-genetic algorithm and/or the MILP model in CPLEX"""

    def __init__(self, inputGraphs: List[str], runsPerGraph: int, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=True,
                 isOptimizedArcSelections=True, isSolvedWithMILP=True, isRace=True):
        """Constructor of a GAvsMILP instance"""
        # Graph solver options
        self.multiRunID = "MultiGAvsMILP--" + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.isSolvedWithGeneticAlg: bool = isSolvedWithGeneticAlg
        self.isSolvedWithMILP: bool = isSolvedWithMILP
        self.isRace: bool = isRace

        # Input graphs attributes
        self.inputGraphs: List[str] = inputGraphs
        self.runsPerGraph: int = runsPerGraph

        # GA Hyperparameters
        self.isOneDimAlphaTable: bool = isOneDimAlphaTable
        self.isOptimizedArcSelections: bool = isOptimizedArcSelections
        self.populationSize: int = 20
        self.numGenerations: int = 40
        self.terminationMethod: str = "setGenerations"
        self.initializationStrategy: str = "perEdge"
        self.initializationDistribution: str = "gaussian"
        self.initializationParams: List[float] = [500.0, 100.0]
        self.selectionMethod: str = "tournament"
        self.tournamentSize: int = 5
        self.crossoverMethod: str = "twoPoint"
        self.crossoverRate: float = 1.0
        self.crossoverAttemptsPerGeneration: int = 1
        self.replacementStrategy: str = "replaceWeakestTwo"
        self.mutationMethod: str = "randomPerEdge"
        self.mutationRate: float = 0.20
        self.perArcEdgeMutationRate: float = 0.25
        self.isDaemonUsed: bool = True
        self.annealingConstant: float = 0.10
        self.daemonStrategy: str = "globalMedian"
        self.daemonStrength: float = 0.10

    def runSolversOnAllGraphs(self) -> None:
        """Solves each graph the specified number of times and writes each run to a CSV"""
        self.createCSV()
        for graphName in self.inputGraphs:
            for runNum in range(self.runsPerGraph):
                print("\n======================================")
                print("GRAPH: " + graphName + "\tRUN: " + str(runNum))
                print("======================================\n")
                gaVSmilp = GAvsMILP(graphName, isSolvedWithGeneticAlg=self.isSolvedWithGeneticAlg,
                                      isOneDimAlphaTable=self.isOneDimAlphaTable,
                                      isOptimizedArcSelections=self.isOptimizedArcSelections,
                                      isSolvedWithMILP=self.isSolvedWithMILP,
                                      isRace=self.isRace,
                                      isDrawing=False,
                                      isLabeling=False,
                                      isGraphing=True,
                                      isOutputtingCPLEX=True)
                # Alpha-GA population attribute & hyperparameters
                gaVSmilp.geneticPop.setPopulationHyperparams(populationSize=self.populationSize,
                                                              numGenerations=self.numGenerations,
                                                              terminationMethod=self.terminationMethod)
                gaVSmilp.geneticPop.setInitializationHyperparams(initializationStrategy=self.initializationStrategy,
                                                                  initializationDistribution=self.initializationDistribution,
                                                                  initializationParams=self.initializationParams)
                gaVSmilp.geneticPop.setIndividualSelectionHyperparams(selectionMethod=self.selectionMethod,
                                                                       tournamentSize=self.tournamentSize)
                gaVSmilp.geneticPop.setCrossoverHyperparams(crossoverMethod=self.crossoverMethod,
                                                             crossoverRate=self.crossoverRate,
                                                             crossoverAttemptsPerGeneration=self.crossoverAttemptsPerGeneration,
                                                             replacementStrategy=self.replacementStrategy)
                gaVSmilp.geneticPop.setMutationHyperparams(mutationMethod=self.mutationMethod,
                                                            mutationRate=self.mutationRate,
                                                            perArcEdgeMutationRate=self.perArcEdgeMutationRate)
                gaVSmilp.geneticPop.setDaemonHyperparams(isDaemonUsed=self.isDaemonUsed,
                                                         annealingConstant=self.annealingConstant,
                                                         daemonStrategy=self.daemonStrategy,
                                                         daemonStrength=self.daemonStrength)
                thisRunData = gaVSmilp.solveGraphWithoutPrints()
                self.writeRowToCSV(thisRunData)

    def createCSV(self) -> None:
        """Creates a CSV file for the output data of the run and writes a header"""
        headerRow = ["Run ID", "Graph Name", "Num Nodes", "Num Sources", "Num Sinks", "Num Edges",
                    "Num Arc Caps", "Target Flow", "is Src/Sink Capped?", "is Src/Sink Charged?", "Pop Size",
                    "Num Gens", "is 1D Alphas?", "is Optimized Arcs?", "termination", "stagnation",
                    "Init Strategy", "Init Dist", "Init Param 0", "Init Param 1", "Selection", "Tourny Size",
                    "Crossover", "CO Rate", "CO Attempts/Gen", "Replacement Strategy", "Mutation", "Mutate Rate",
                    "Per Arc/Edge Mutate Rate", "is Daemon Used?", "Annealing Constant", "Daemon Strategy",
                    "Daemon Strength", "GA Best Obj Val", "GA Runtime (sec)", "MILP Obj Val",
                    "MILP Runtime (sec)", "Time Limit", "Status", "Status Code", "Best Bound",
                    "MILP Gap", "GA Gap", "MILP Gap - GA GAP"]
        # Build Output Header
        outputHeader = [["MULTI-GA vs. MILP RESULTS", self.multiRunID], headerRow]
        # Create CSV File
        currDir = os.getcwd()
        csvName = self.multiRunID + ".csv"
        catPath = os.path.join(currDir, csvName)
        csvFile = open(catPath, "w+", newline="")
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(outputHeader)

    def writeRowToCSV(self, outputRow: list) -> None:
        """Appends the most recent data onto a .csv file"""
        currDir = os.getcwd()
        csvName = self.multiRunID + ".csv"
        catPath = os.path.join(currDir, csvName)
        csvFile = open(catPath, "a", newline="")
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(outputRow)
