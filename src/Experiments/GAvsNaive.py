import csv
import os
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt

from src.AlphaGenetic.Population import Population
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph


class GAvsNaive:
    """Class that solves a single graph using the alpha-genetic algorithm and/or the naive hillclimb metahueristic"""

    def __init__(self, inputGraphName: str, isSolvedWithGeneticAlg=True, isOneDimAlphaTable=True,
                 isOptimizedArcSelections=True, isSolvedWithNaive=True, isDrawing=True, isLabeling=True,
                 isGraphing=True):
        """Constructor of a GAvsNaive instance"""
        # Graph solver options
        self.runID = "GAvsNaive--" + inputGraphName + "--" + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.isSolvedWithGeneticAlg: bool = isSolvedWithGeneticAlg
        self.isSolvedWithNaive: bool = isSolvedWithNaive
        self.isDrawing: bool = isDrawing
        self.isLabeling: bool = isLabeling
        self.isGraphing: bool = isGraphing
        if self.isGraphing is True:
            matplotlib.use("agg")  # Simpler MatPlotLib backend for rendering high number of PNGs per run

        # Input graph attributes
        self.graphName: str = inputGraphName
        self.graph: CandidateGraph = CandidateGraph()
        self.graph = self.graph.loadCandidateGraph(self.graphName + ".p")
        self.minTargetFlow: float = self.graph.totalPossibleDemand

        # Alpha-GA population attribute & hyperparameters
        self.geneticPop: Population = Population(self.graph, self.minTargetFlow,
                                                 isOneDimAlphaTable=isOneDimAlphaTable,
                                                 isOptimizedArcSelections=isOptimizedArcSelections)
        self.geneticPop.setPopulationHyperparams(populationSize=20,
                                                 numGenerations=50,
                                                 terminationMethod="setGenerations")
        self.geneticPop.setInitializationHyperparams(initializationStrategy="perEdge",
                                                 initializationDistribution="gaussian",
                                                 initializationParams=[500.0, 100.0])
        self.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                            tournamentSize=4)
        self.geneticPop.setCrossoverHyperparams(crossoverMethod="onePoint",
                                                crossoverRate=1.0,
                                                crossoverAttemptsPerGeneration=1,
                                                replacementStrategy="replaceWeakestTwo")
        self.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                               mutationRate=0.10,
                                               perArcEdgeMutationRate=0.25)
        self.geneticPop.setDaemonHyperparams(isDaemonUsed=True, annealingConstant=0.10,
                                                      daemonStrategy="globalMean", daemonStrength=0.10)
        self.gaSolution = None

        # Naive hillclimb attribute
        self.naivePop: Population = Population(self.graph, self.minTargetFlow,
                                                 isOneDimAlphaTable=isOneDimAlphaTable,
                                                 isOptimizedArcSelections=isOptimizedArcSelections)
        self.geneticPop.setPopulationHyperparams(populationSize=20,
                                                 numGenerations=50,
                                                 terminationMethod="setGenerations")
        self.naiveSolution = None

        # Execution and output attributes
        self.gaRuntimeInSeconds: float = -1
        self.naiveRuntimeInSeconds: float = -1

    def solveGraph(self) -> None:
        """Solves the graph with the genetic algorithm and/or the naive hillclimb population"""
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
            gaFinishTime = datetime.now()
            gaRuntime = gaFinishTime - gaStartTime
            self.gaRuntimeInSeconds = gaRuntime.seconds + gaRuntime.microseconds / 1000000
            print("\nGA Runtime (in seconds): " + str(self.gaRuntimeInSeconds))
        # Solve the naive hillclimb population
        if self.isSolvedWithNaive is True:
            print("\n============================================================================")
            # Timestamp and start the HC
            print("\n\nSolving the " + self.graphName + " graph with a naive-HC population of " +
                  str(self.geneticPop.populationSize) + " for " + str(self.geneticPop.numGenerations) + " generations...\n")
            naiveStartTime = datetime.now()
            self.naiveSolution = self.naivePop.solveWithNaiveHillClimb(printGenerations=True, drawing=self.isDrawing,
                                                               drawLabels=self.isLabeling, isGraphing=self.isGraphing,
                                                               runID=self.runID)
            print("\nGenetic Algorithm Complete!!!\nBest Solution Found = " + str(self.gaSolution.trueCost))
            # Draw if expected
            if self.isDrawing is True:
                naiveVis = SolutionVisualizer(self.naiveSolution)
                if self.isLabeling is True:
                    naiveVis.drawLabeledSolution(leadingText="GA-BEST_")
                else:
                    naiveVis.drawUnlabeledSolution(leadingText="GA-BEST_")
            # Timestamp and stop the GA
            naiveFinishTime = datetime.now()
            naiveRuntime = naiveFinishTime - naiveStartTime
            self.naiveRuntimeInSeconds = naiveRuntime.seconds + naiveRuntime.microseconds / 1000000
            print("\nNaive HC Runtime (in seconds): " + str(self.naiveRuntimeInSeconds))
        if self.isSolvedWithGeneticAlg is True and self.isSolvedWithNaive is True and self.isGraphing is True:
            self.plotRuntimeConvergenceAgainstNaive()
        # self.saveOutputAsCSV()  # TODO - UNCOMMENT AFTER REVISING CSV FORMATTING
        print("\nRun complete!\n")

    def plotRuntimeConvergenceAgainstNaive(self) -> None:
        """Plots the convergence graph for the GA and HC populations"""
        # Get plt figure
        fig = plt.figure()
        ax = fig.add_subplot()
        # Plot all data
        ax.plot(self.geneticPop.generationTimestamps, self.geneticPop.convergenceStats, label="Most Fit GA Ind", color="g")
        ax.plot(self.geneticPop.generationTimestamps, self.geneticPop.meanStats, label="Mean GA Fitness", color="b", linestyle="--")
        ax.plot(self.geneticPop.generationTimestamps, self.geneticPop.medianStats, label="Median GA Fitness", color="c", linestyle="--")
        ax.plot(self.naivePop.generationTimestamps, self.naivePop.convergenceStats, label="Most Fit HC Ind", color="r")
        ax.plot(self.naivePop.generationTimestamps, self.naivePop.meanStats, label="Mean HC Fitness", color="y", linestyle=":")
        ax.plot(self.naivePop.generationTimestamps, self.naivePop.medianStats, label="Median HC Fitness", color="m", linestyle=":")
        # Add graph elements
        ax.set_title("GA Convergence Against Naive HC")
        ax.legend(loc=1)
        ax.set_ylabel("Obj. Value")
        ax.set_xlabel("Runtime (in sec)")
        # Save timestamped plot
        plt.savefig(self.runID + ".png")
        plt.close(fig)


    # TODO - REVISE HERE DOWN FOR OUTPUTTING CSV FILE
    def saveOutputAsCSV(self) -> None:
        """Writes all of the output data to disc as a CSV file"""
        print("\nWriting output to disc as '" + self.runID + ".csv'...")
        self.createCSV()
        if self.isSolvedWithGeneticAlg is True:
            self.writeRowToCSV(self.buildGAHeader())
            self.writeRowToCSV(self.buildGAData())
            self.writeRowToCSV([])
            self.writeRowToCSV(["GA Generation Timestamps"])
            self.writeRowToCSV(self.geneticPop.generationTimestamps)
            self.writeRowToCSV(["Most Fit Ind."])
            self.writeRowToCSV(self.geneticPop.convergenceStats)
            self.writeRowToCSV(["Mean Fitness"])
            self.writeRowToCSV(self.geneticPop.meanStats)
            self.writeRowToCSV(["Median Fitness"])
            self.writeRowToCSV(self.geneticPop.medianStats)
            self.writeRowToCSV(["Std Dev"])
            self.writeRowToCSV(self.geneticPop.stdDevStats)
            self.writeRowToCSV([])
        if self.isSolvedWithMILP is True:
            self.writeRowToCSV(self.buildMILPHeaderRow())
            self.writeRowToCSV(self.buildMILPDataRow())
            self.writeRowToCSV(["MILP Solver Timestamps"])
            self.writeRowToCSV(self.milpCplexSolver.runtimeTimestamps)
            self.writeRowToCSV(["MILP Runtime Obj Values"])
            self.writeRowToCSV(self.milpCplexSolver.runtimeObjectiveValues)
            self.writeRowToCSV(["MILP Runtime Bound"])
            self.writeRowToCSV(self.milpCplexSolver.runtimeBestBounds)
            self.writeRowToCSV(["MILP Runtime Gap"])
            self.writeRowToCSV(self.milpCplexSolver.runtimeGaps)

    def createCSV(self) -> None:
        """Creates a CSV file for the output data of the run and writes a header"""
        # Build Output Header
        outputHeader = [["GA vs. MILP RESULTS", self.runID],
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
                "Per Arc/Edge Mutate Rate", "is Daemon Used?", "Annealing Constant", "Daemon Strategy",
                "Daemon Strength", "GA Best Obj Val", "GA Runtime (sec)"]

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
                self.geneticPop.isDaemonUsed, self.geneticPop.annealingConstant, self.geneticPop.daemonStrategy,
                self.geneticPop.daemonStrength, self.gaSolution.trueCost, self.geneticRuntimeInSeconds]

    @staticmethod
    def buildNaiveHeaderRow() -> list:
        """Builds a list containing the solution detail headers of the MILP formulation in CPLEX"""
        return ["MILP Obj Val", "MILP Runtime (sec)", "Time Limit", "Status", "Status Code", "Best Bound",
                "MILP Gap", "GA Gap", "MILP Gap - GA GAP"]

    def buildNaiveDataRow(self) -> list:
        """Builds a list containing the solution details of the MILP formulation in CPLEX"""
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
        if self.isSolvedWithMILP is True:
            if self.isRace is True:
                self.milpCplexSolver.setTimeLimit(self.geneticRuntimeInSeconds)
            self.milpCplexSolver.findSolution(printDetails=False)
            print("CPLEX Complete!")
        if self.isSolvedWithGeneticAlg is True and self.isSolvedWithMILP is True and self.isRace is True and self.isGraphing is True:
            self.plotRuntimeConvergenceAgainstMILP()
        return self.buildSingleRowRunData()

    def buildSingleRowRunHeaders(self) -> list:
        """Builds the headers for a single row containing the data of the run"""
        headerRow = ["Run ID", "Graph Name", "Num Nodes", "Num Sources", "Num Sinks", "Num Edges",
         "Num Arc Caps", "Target Flow", "is Src/Sink Capped?", "is Src/Sink Charged?"]
        if self.isSolvedWithGeneticAlg is True:
            headerRow.extend(self.buildGAHeader())
        if self.isSolvedWithMILP is True:
            headerRow.extend(self.buildMILPHeaderRow())
        return headerRow

    def buildSingleRowRunData(self) -> list:
        """Builds a single row containing the data of the run"""
        dataRow = [self.runID, self.graphName, self.graph.numTotalNodes, self.graph.numSources, self.graph.numSinks,
                         self.graph.numEdges, self.graph.numArcsPerEdge, self.minTargetFlow,
                         self.graph.isSourceSinkCapacitated, self.graph.isSourceSinkCharged]
        if self.isSolvedWithGeneticAlg is True:
            dataRow.extend(self.buildGAData())
        if self.isSolvedWithMILP is True:
            dataRow.extend(self.buildMILPDataRow())
        return dataRow
