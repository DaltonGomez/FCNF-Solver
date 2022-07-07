import csv
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from src.AlphaGenetic.Population import Population
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX


class GAvsCPLEX:
    """Class that solves a single graph using the alpha-genetic algorithm and/or the MILP model in CPLEX"""

    def __init__(self, inputGraphName: str, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=True,
                 isOptimizedArcSelections=True, isSolvedWithCPLEX=True, isRace=True,
                 isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=True):
        """Constructor of a GAvsCPLEX instance"""
        # Graph solver options
        self.runID = "GAvsCPLEX--" + inputGraphName + "--" + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.isSolvedWithGeneticAlg: bool = isSolvedWithGeneticAlg
        self.isSolvedWithCPLEX: bool = isSolvedWithCPLEX
        self.isRace: bool = isRace
        self.isDrawing: bool = isDrawing
        self.isLabeling: bool = isLabeling
        self.isGraphing: bool = isGraphing
        self.isOutputtingCPLEX: bool = isOutputtingCPLEX

        # Input graph attributes
        self.graphName: str = inputGraphName
        self.graph: CandidateGraph = CandidateGraph()
        self.graph = self.graph.loadCandidateGraph(self.graphName + ".p")
        self.minTargetFlow: float = self.graph.totalPossibleDemand

        # Alpha-GA population attribute & hyperparameters
        self.geneticPop: Population = Population(self.graph, self.minTargetFlow,
                         isOneDimAlphaTable=isOneDimAlphaTable, isOptimizedArcSelections=isOptimizedArcSelections)
        self.geneticPop.setPopulationHyperparams(populationSize=10,
                                                 numGenerations=10,
                                                 terminationMethod="setGenerations",
                                                 stagnationPeriod=5,
                                                 isOneDimAlphaTable=isOneDimAlphaTable,
                                                 isOptimizedArcSelections=isOptimizedArcSelections
                                                 )
        self.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                 initializationDistribution="digital",
                                                 initializationParams=[0.0, 100000.0])
        self.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                            tournamentSize=4)
        self.geneticPop.setCrossoverHyperparams(crossoverMethod="onePoint",
                                                crossoverRate=1.0,
                                                crossoverAttemptsPerGeneration=1,
                                                replacementStrategy="replaceWeakestTwo")
        self.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                               mutationRate=0.05,
                                               perArcEdgeMutationRate=0.25)
        self.gaSolution = None

        # MILP CPLEX attribute
        self.milpCplexSolver: MILPsolverCPLEX = MILPsolverCPLEX(self.graph, self.minTargetFlow,
                                          isOneArcPerEdge=False, logOutput=isOutputtingCPLEX)
        self.milpSolution = None

        # Execution and output attributes
        self.geneticRuntimeInSeconds: float = -1

    def solveGraph(self) -> None:
        """Solves the graph with the genetic algorithm and/or the MILP formulation in CPLEX"""
        # Solve the alpha-GA population
        if self.isSolvedWithGeneticAlg is True:
            # Timestamp and start the GA
            print("\n\nSolving the " + self.graphName + " graph with a GA population of " + str(
                self.geneticPop.populationSize) + " for " + str(self.geneticPop.numGenerations) + " generations...\n")
            gaStartTime = datetime.now()
            # Evolve the Alpha-GA population
            self.gaSolution = self.geneticPop.evolvePopulation(printGenerations=True, drawing=self.isDrawing,
                                                          drawLabels=self.isLabeling, isGraphing=self.isGraphing,
                                                          runID=self.runID)
            print("\nGenetic Algorithm Complete!!!\nBest Solution Found = " + str(self.gaSolution.trueCost))
            # Draw if expected
            if self.isDrawing is True:
                gaVis = SolutionVisualizer(self.gaSolution)
                if self.isLabeling is True:
                    gaVis.drawLabeledSolution(leadingText="GA-BEST_")
                else:
                    gaVis.drawUnlabeledSolution(leadingText="GA-BEST_")
            # Timestamp and stop the GA
            gaFinishOptStart = datetime.now()
            gaRuntime = gaFinishOptStart - gaStartTime
            self.geneticRuntimeInSeconds = gaRuntime.seconds + gaRuntime.microseconds/1000000
            print("\nGA Runtime (in seconds): " + str(self.geneticRuntimeInSeconds))
        # Solve the MILP formulation in CPLEX
        if self.isSolvedWithCPLEX is True:
            print("\n============================================================================")
            print("Solving the " + self.graphName + " graph with a MILP formulation in CPLEX...\n")
            # Set time limit if CPLEX is racing GA
            if self.isRace is True:
                self.milpCplexSolver.setTimeLimit(self.geneticRuntimeInSeconds)
            # Call CPLEX to solve MILP
            self.milpCplexSolver.findSolution(printDetails=False)
            print("\nCPLEX MILP Solver Complete!!!\nBest Solution Found = " + str(self.milpCplexSolver.getObjectiveValue()))
            # Draw if expected
            if self.isDrawing is True:
                print("\nFLAGGING ANY KEY ERRORS FROM CPLEX...")
                self.milpSolution = self.milpCplexSolver.writeSolution()
                milpVis = SolutionVisualizer(self.milpSolution)
                if self.isLabeling is True:
                    milpVis.drawLabeledSolution(leadingText="MILP_")
                else:
                    milpVis.drawUnlabeledSolution(leadingText="MILP_")
            # Print solution details
            print("\nCPLEX MILP Objective Value: " + str(self.milpCplexSolver.getObjectiveValue()))
            print("CPLEX Runtime (in seconds): " + str(self.milpCplexSolver.getCplexRuntime()))
            print("CPLEX Status " + self.milpCplexSolver.getCplexStatus())
            print("CPLEX Gap: " + str(self.milpCplexSolver.getGap() * 100) + "%")
            print("CPLEX Best Bound: " + str(self.milpCplexSolver.getBestBound()))
        if self.isSolvedWithGeneticAlg is True and self.isSolvedWithCPLEX is True and self.isRace is True and self.isGraphing is True:
            self.plotConvergenceAgainstCPLEX()
        self.saveOutputAsCSV()
        print("\nRun complete!\n")

    def plotConvergenceAgainstCPLEX(self) -> None:
        """Plots the convergence graph against CPLEX's best found solution and gap/best bound"""
        # Get generations, CPLEX data and plt figure
        numGenerations = len(self.geneticPop.convergenceStats)
        generations = list(range(numGenerations))
        cplexObjectiveValue = self.milpCplexSolver.getObjectiveValue()
        cplexBestBound = self.milpCplexSolver.getBestBound()
        fig = plt.figure()
        ax = fig.add_subplot()
        # Plot all data
        ax.plot(generations, self.geneticPop.convergenceStats, label="Most Fit Individual", color="g")
        ax.plot(generations, self.geneticPop.meanStats, label="Mean Pop. Fitness", color="b")
        ax.plot(generations, self.geneticPop.medianStats, label="Median Pop. Fitness", color="c")
        ax.plot(generations, np.full(numGenerations, cplexObjectiveValue), label="CPLEX Best Soln", linestyle="--", color="y")
        ax.plot(generations, np.full(numGenerations, cplexBestBound), label="CPLEX MILP Bound", linestyle=":", color="r")
        # Add graph elements
        ax.set_title("GA Convergence Against CPLEX over Equal Runtime")
        ax.legend(loc=4)
        ax.set_ylim(ymin=0, ymax=max(cplexObjectiveValue, max(self.geneticPop.meanStats))*1.25)
        ax.set_ylabel("Obj. Value")
        ax.set_xlabel("Runtime")
        # Save timestamped plot
        plt.savefig(self.runID + ".png")
        plt.close(fig)

    def saveOutputAsCSV(self) -> None:
        """Writes all of the output data to disc as a CSV file"""
        print("\nWriting output to disc as '" + self.runID + ".csv'...")
        self.createCSV()
        if self.isSolvedWithGeneticAlg is True:
            self.writeRowToCSV(self.buildGAHeader())
            self.writeRowToCSV(self.buildGAData())
            self.writeRowToCSV([])
            self.writeRowToCSV(["Most Fit Ind."])
            self.writeRowToCSV(self.geneticPop.convergenceStats)
            self.writeRowToCSV(["Mean Fitness"])
            self.writeRowToCSV(self.geneticPop.meanStats)
            self.writeRowToCSV(["Median Fitness"])
            self.writeRowToCSV(self.geneticPop.medianStats)
            self.writeRowToCSV(["Std Dev"])
            self.writeRowToCSV(self.geneticPop.stdDevStats)
            self.writeRowToCSV([])
        if self.isSolvedWithCPLEX is True:
            self.writeRowToCSV(self.buildCPLEXHeaderRow())
            self.writeRowToCSV(self.buildCPLEXDataRow())

    def createCSV(self) -> None:
        """Creates a CSV file for the output data of the run and writes a header"""
        # Build Output Header
        outputHeader = [["GA vs. CPLEX RESULTS OUTPUT", self.runID],
                        [],
                        ["INPUT GRAPH DATA"],
                        ["Graph Name", "Num Nodes", "Num Sources", "Num Sinks", "Num Edges",
                         "Num Arc Caps", "Target Flow", "is Src/Sink Capped?", "is Src/Sink Charged?"],
                        [self.graphName, self.graph.numTotalNodes, self.graph.numSources, self.graph.numSinks,
                         self.graph.numEdges, self.graph.numArcsPerEdge, self.minTargetFlow,
                         self.graph.isSourceSinkCapacitated, self.graph.isSourceSinkCharged],
                        []
                        ]
        # Create CSV File
        currDir = os.getcwd()
        csvName = self.runID + ".csv"
        catPath = os.path.join(currDir, csvName)
        csvFile = open(catPath, "w+", newline="")
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(outputHeader)

    def writeRowToCSV(self, outputRow: list) -> None:
        """Appends the most recent data onto a .csv file"""
        currDir = os.getcwd()
        csvName = self.runID + ".csv"
        catPath = os.path.join(currDir, csvName)
        csvFile = open(catPath, "a", newline="")
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(outputRow)

    @staticmethod
    def buildGAHeader() -> list:
        """Builds a list containing the population's hyperparameter headers for exporting to a CSV"""
        return ["Pop Size", "Num Gens", "is 1D Alphas?", "is Optimized Arcs?", "termination", "stagnation",
                "Init Strategy", "Init Dist", "Init Param 0", "Init Param 1", "Selection", "Tourny Size",
                "Crossover", "CO Rate", "CO Attempts/Gen", "Replacement Strategy", "Mutation", "Mutate Rate",
                "Per Arc/Edge Mutate Rate", "GA Best Obj Val", "GA Runtime (sec)"]

    def buildGAData(self) -> list:
        """Builds a list containing the population's hyperparameters for exporting to a CSV"""
        return [self.geneticPop.populationSize, self.geneticPop.numGenerations, self.geneticPop.isOneDimAlphaTable,
                self.geneticPop.isOptimizedArcSelections, self.geneticPop.terminationMethod,
                self.geneticPop.stagnationPeriod, self.geneticPop.initializationStrategy,
                self.geneticPop.initializationDistribution, self.geneticPop.initializationParams[0],
                self.geneticPop.initializationParams[1], self.geneticPop.selectionMethod,
                self.geneticPop.tournamentSize, self.geneticPop.crossoverMethod, self.geneticPop.crossoverRate,
                self.geneticPop.crossoverAttemptsPerGeneration, self.geneticPop.replacementStrategy,
                self.geneticPop.mutationMethod, self.geneticPop.mutationRate, self.geneticPop.perArcEdgeMutationRate,
                self.gaSolution.trueCost, self.geneticRuntimeInSeconds]

    @staticmethod
    def buildCPLEXHeaderRow() -> list:
        """Builds a list containing the solution detail headers of the CPLEX solver on the MILP formulation"""
        return ["CPLEX Obj Val", "CPLEX Runtime (sec)", "Time Limit", "Status", "Status Code", "Best Bound",
                "MILP Gap", "GA Gap", "MILP Gap - GA GAP"]

    def buildCPLEXDataRow(self) -> list:
        """Builds a list containing the solution details of the CPLEX solver on the MILP formulation"""
        gaGap = 1 - self.milpCplexSolver.getBestBound()/self.gaSolution.trueCost
        return [self.milpCplexSolver.getObjectiveValue(), self.milpCplexSolver.getCplexRuntime(),
                self.milpCplexSolver.getTimeLimit(), self.milpCplexSolver.getCplexStatus(),
                self.milpCplexSolver.getCplexStatusCode(), self.milpCplexSolver.getBestBound(),
                self.milpCplexSolver.getGap(), gaGap, self.milpCplexSolver.getGap() - gaGap]

    def solveGraphWithoutPrints(self) -> list:
        """Solves the graph using the GA vs. CPLEX method but without printing (for use in multi-run experiments)"""
        # Solve the alpha-GA population
        if self.isSolvedWithGeneticAlg is True:
            gaStartTime = datetime.now()
            self.gaSolution = self.geneticPop.evolvePopulation(printGenerations=False, drawing=False,
                                                               drawLabels=False, isGraphing=self.isGraphing,
                                                               runID=self.runID)
            gaFinishOptStart = datetime.now()
            gaRuntime = gaFinishOptStart - gaStartTime
            self.geneticRuntimeInSeconds = gaRuntime.seconds + gaRuntime.microseconds / 1000000
            print("GA Complete!")
        # Solve the MILP formulation in CPLEX
        if self.isSolvedWithCPLEX is True:
            if self.isRace is True:
                self.milpCplexSolver.setTimeLimit(self.geneticRuntimeInSeconds)
            self.milpCplexSolver.findSolution(printDetails=False)
            print("CPLEX Complete!")
        if self.isSolvedWithGeneticAlg is True and self.isSolvedWithCPLEX is True and self.isRace is True and self.isGraphing is True:
            self.plotConvergenceAgainstCPLEX()
        return self.buildSingleRowRunData()

    def buildSingleRowRunHeaders(self) -> list:
        """Builds the headers for a single row containing the data of the run"""
        headerRow = ["Run ID", "Graph Name", "Num Nodes", "Num Sources", "Num Sinks", "Num Edges",
         "Num Arc Caps", "Target Flow", "is Src/Sink Capped?", "is Src/Sink Charged?"]
        if self.isSolvedWithGeneticAlg is True:
            headerRow.extend(self.buildGAHeader())
        if self.isSolvedWithCPLEX is True:
            headerRow.extend(self.buildCPLEXHeaderRow())
        return headerRow

    def buildSingleRowRunData(self) -> list:
        """Builds a single row containing the data of the run"""
        dataRow = [self.runID, self.graphName, self.graph.numTotalNodes, self.graph.numSources, self.graph.numSinks,
                         self.graph.numEdges, self.graph.numArcsPerEdge, self.minTargetFlow,
                         self.graph.isSourceSinkCapacitated, self.graph.isSourceSinkCharged]
        if self.isSolvedWithGeneticAlg is True:
            dataRow.extend(self.buildGAData())
        if self.isSolvedWithCPLEX is True:
            dataRow.extend(self.buildCPLEXDataRow())
        return dataRow
