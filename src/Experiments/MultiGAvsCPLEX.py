import csv
import os
from datetime import datetime
from typing import List

from src.Experiments.GAvsCPLEX import GAvsCPLEX


class MultiGAvsCPLEX:
    """Class that solves a multiple graphs using the alpha-genetic algorithm and/or the MILP model in CPLEX"""

    def __init__(self, inputGraphs: List[str], runsPerGraph: int, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=True,
                 isOptimizedArcSelections=True, isSolvedWithCPLEX=True, isRace=True):
        """Constructor of a GAvsCPLEX instance"""
        # Graph solver options
        self.multiRunID = "MultiGAvsCPLEX--" + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.isSolvedWithGeneticAlg: bool = isSolvedWithGeneticAlg
        self.isSolvedWithCPLEX: bool = isSolvedWithCPLEX
        self.isRace: bool = isRace

        # Input graphs attributes
        self.inputGraphs: List[str] = inputGraphs
        self.runsPerGraph: int = runsPerGraph

        # GA Hyperparameters
        self.isOneDimAlphaTable: bool = isOneDimAlphaTable
        self.isOptimizedArcSelections: bool = isOptimizedArcSelections
        self.populationSize: int = 10
        self.numGenerations: int = 10
        self.terminationMethod: str = "setGenerations"
        self.stagnationPeriod: int = 5
        self.initializationStrategy: str = "perEdge"
        self.initializationDistribution: str = "digital"
        self.initializationParams: List[float] = [0.0, 100000.0]
        self.selectionMethod: str = "tournament"
        self.tournamentSize: int = 4
        self.crossoverMethod: str = "onePoint"
        self.crossoverRate: float = 1.0
        self.crossoverAttemptsPerGeneration: int = 1
        self.replacementStrategy: str = "replaceWeakestTwo"
        self.mutationMethod: str = "randomPerEdge"
        self.mutationRate: float = 0.05
        self.perArcEdgeMutationRate: float = 0.25

        # Output attributes
        self.isCsvCreated = False

    def runSolversOnAllGraphs(self) -> None:
        """Solves each graph the specified number of times and writes each run to a CSV"""
        for graphName in self.inputGraphs:
            for runNum in range(self.runsPerGraph):
                print("\n======================================")
                print("GRAPH: " + graphName + "\tRUN: " + str(runNum))
                print("======================================\n")
                gaVScplex = GAvsCPLEX(graphName, isSolvedWithGeneticAlg=self.isSolvedWithGeneticAlg,
                                      isOneDimAlphaTable=self.isOneDimAlphaTable,
                                      isOptimizedArcSelections=self.isOptimizedArcSelections,
                                      isSolvedWithCPLEX=self.isSolvedWithCPLEX,
                                      isRace=self.isRace,
                                      isDrawing=False,
                                      isLabeling=False,
                                      isGraphing=False,
                                      isOutputtingCPLEX=False)
                if self.isCsvCreated is False:
                    headerRow = gaVScplex.buildSingleRowRunHeaders()
                    self.createCSV(headerRow)
                    self.isCsvCreated = True
                thisRunData = gaVScplex.solveGraphWithoutPrints()
                self.writeRowToCSV(thisRunData)

    def createCSV(self, headerRow: list) -> None:
        """Creates a CSV file for the output data of the run and writes a header"""
        # Build Output Header
        outputHeader = [["MULTI-GA vs. CPLEX RESULTS OUTPUT", self.multiRunID], headerRow]
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