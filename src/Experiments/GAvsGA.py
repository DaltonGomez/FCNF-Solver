import csv
import os
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt

from src.AlphaGenetic.Population import Population
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph


class GAvsGA:
    """Class that solves a single graph using the two alpha-genetic populations for comparison"""

    def __init__(self, inputGraphName: str, isPop1OneDimAlpha=True, isPop1ArcOptimized=True,
                 isPop2OneDimAlpha=False, isPop2ArcOptimized=True, isDrawing=True, isLabeling=True, isGraphing=True):
        """Constructor of a GAvsGA instance"""
        # Graph solver options
        self.runID = "GAvsGA--" + inputGraphName + "--" + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
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

        # Alpha-GA population one attribute & hyperparameters
        self.geneticPopOne: Population = Population(self.graph, self.minTargetFlow,
                         isOneDimAlphaTable=isPop1OneDimAlpha, isOptimizedArcSelections=isPop1ArcOptimized)
        self.geneticPopOne.setPopulationHyperparams(populationSize=10,
                                                 numGenerations=10,
                                                 terminationMethod="setGenerations")
        self.geneticPopOne.setInitializationHyperparams(initializationStrategy="perEdge",
                                                 initializationDistribution="digital",
                                                 initializationParams=[5.0, 100000.0])
        self.geneticPopOne.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                            tournamentSize=4)
        self.geneticPopOne.setCrossoverHyperparams(crossoverMethod="onePoint",
                                                crossoverRate=1.0,
                                                crossoverAttemptsPerGeneration=1,
                                                replacementStrategy="replaceWeakestTwo")
        self.geneticPopOne.setMutationHyperparams(mutationMethod="randomPerEdge",
                                               mutationRate=0.05,
                                               perArcEdgeMutationRate=0.25)
        self.geneticPopOne.setDaemonHyperparams(isDaemonUsed=True,
                                                annealingConstant=0.5,
                                                daemonStrategy="globalMean",
                                                daemonStrength=1)
        self.gaSolutionOne = None

        # Alpha-GA population two attribute & hyperparameters
        self.geneticPopTwo: Population = Population(self.graph, self.minTargetFlow, isOneDimAlphaTable=isPop2OneDimAlpha,
                                                    isOptimizedArcSelections=isPop2ArcOptimized)
        self.geneticPopTwo.setPopulationHyperparams(populationSize=10,
                                                    numGenerations=10,
                                                    terminationMethod="setGenerations")
        self.geneticPopTwo.setInitializationHyperparams(initializationStrategy="perArc",
                                                        initializationDistribution="digital",
                                                        initializationParams=[5.0, 100000.0])
        self.geneticPopTwo.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                             tournamentSize=4)
        self.geneticPopTwo.setCrossoverHyperparams(crossoverMethod="onePoint",
                                                   crossoverRate=1.0,
                                                   crossoverAttemptsPerGeneration=1,
                                                   replacementStrategy="replaceWeakestTwo")
        self.geneticPopTwo.setMutationHyperparams(mutationMethod="randomPerEdge",
                                                  mutationRate=0.05,
                                                  perArcEdgeMutationRate=0.25)
        self.geneticPopTwo.setDaemonHyperparams(isDaemonUsed=True,
                                                annealingConstant=0.5,
                                                daemonStrategy="globalMean",
                                                daemonStrength=1)
        self.gaSolutionTwo = None

    def solveGraph(self) -> None:
        """Solves the graph with the two genetic algorithm populations"""
        # Solve first GA Population
        print("\n\nSolving the " + self.graphName + " graph with GA Population #1 of " +
              str(self.geneticPopOne.populationSize) + " for " + str(self.geneticPopOne.numGenerations) +
              " generations...\n")
        # Evolve the Alpha-GA population #1
        self.gaSolutionOne = self.geneticPopOne.evolvePopulation(printGenerations=True, drawing=self.isDrawing,
                                                                 drawLabels=self.isLabeling, isGraphing=self.isGraphing,
                                                                 runID=self.runID + "--POP1")
        print("\nGenetic Algorithm Population #1 Complete!!!\nBest Solution Found = " + str(self.gaSolutionOne.trueCost))
        # Draw if expected
        if self.isDrawing is True:
            gaOneVis = SolutionVisualizer(self.gaSolutionOne)
            if self.isLabeling is True:
                gaOneVis.drawLabeledSolution(leadingText="GA1-BEST_")
            else:
                gaOneVis.drawUnlabeledSolution(leadingText="GA1-BEST_")

        # Solve second GA Population
        print("\n\nSolving the " + self.graphName + " graph with GA Population #2 of " +
              str(self.geneticPopTwo.populationSize) + " for " + str(self.geneticPopTwo.numGenerations) +
              " generations...\n")
        # Evolve the Alpha-GA population #2
        self.gaSolutionTwo = self.geneticPopTwo.evolvePopulation(printGenerations=True, drawing=self.isDrawing,
                                                                 drawLabels=self.isLabeling,
                                                                 isGraphing=self.isGraphing, runID=self.runID + "--POP2")
        print("\nGenetic Algorithm Population #2 Complete!!!\nBest Solution Found = " + str(self.gaSolutionTwo.trueCost))
        # Draw if expected
        if self.isDrawing is True:
            gaTwoVis = SolutionVisualizer(self.gaSolutionTwo)
            if self.isLabeling is True:
                gaTwoVis.drawLabeledSolution(leadingText="GA2-BEST_")
            else:
                gaTwoVis.drawUnlabeledSolution(leadingText="GA2-BEST_")
        if self.isGraphing is True:
            self.plotGeneticConvergenceAgainstAnother()
        self.saveOutputAsCSV()
        print("\nRun complete!\n")

    def plotGeneticConvergenceAgainstAnother(self) -> None:
        """Plots the convergence graph of the two alpha GA solutions"""
        gaOneTimestamps = self.geneticPopOne.generationTimestamps
        gaTwoTimestamps = self.geneticPopTwo.generationTimestamps
        fig = plt.figure()
        ax = fig.add_subplot()
        # Plot all data
        ax.plot(gaOneTimestamps, self.geneticPopOne.convergenceStats, label="GA 1 Most Fit", color="b")
        ax.plot(gaTwoTimestamps, self.geneticPopTwo.convergenceStats, label="GA 2 Most Fit", color="c")
        # ax.plot(gaOneTimestamps, self.geneticPopOne.meanStats, label="GA 1 Mean", color="r", linestyle="--")
        # ax.plot(gaTwoTimestamps, self.geneticPopTwo.meanStats, label="GA 2 Mean", color="m", linestyle="--")
        ax.plot(gaOneTimestamps, self.geneticPopOne.medianStats, label="GA 1 Median", color="g", linestyle=":")
        ax.plot(gaTwoTimestamps, self.geneticPopTwo.medianStats, label="GA 2 Median", color="y", linestyle=":")
        # Add graph elements
        ax.set_title("GA Convergence Against One Another")
        ax.legend(loc=1)
        ax.set_ylabel("Obj. Value")
        ax.set_xlabel("Runtime (in seconds)")
        # Save timestamped plot
        plt.savefig(self.runID + ".png")
        plt.close(fig)

    def saveOutputAsCSV(self) -> None:
        """Writes all of the output data to disc as a CSV file"""
        print("\nWriting output to disc as '" + self.runID + ".csv'...")
        self.createCSV()
        self.writeRowToCSV(["POPULATION ONE"])
        self.writeRowToCSV(self.buildGAHeader())
        self.writeRowToCSV(self.buildGA1Data())
        self.writeRowToCSV(["POP1 Timestamps"])
        self.writeRowToCSV(self.geneticPopOne.generationTimestamps)
        self.writeRowToCSV(["POP1 Most Fit Ind."])
        self.writeRowToCSV(self.geneticPopOne.convergenceStats)
        self.writeRowToCSV(["POP1 Mean Fitness"])
        self.writeRowToCSV(self.geneticPopOne.meanStats)
        self.writeRowToCSV(["POP1 Median Fitness"])
        self.writeRowToCSV(self.geneticPopOne.medianStats)
        self.writeRowToCSV(["POP1 Std Dev"])
        self.writeRowToCSV(self.geneticPopOne.stdDevStats)
        self.writeRowToCSV([])
        self.writeRowToCSV(["POPULATION TWO"])
        self.writeRowToCSV(self.buildGAHeader())
        self.writeRowToCSV(self.buildGA2Data())
        self.writeRowToCSV(["POP2 Timestamps"])
        self.writeRowToCSV(self.geneticPopTwo.generationTimestamps)
        self.writeRowToCSV(["POP2 Most Fit Ind."])
        self.writeRowToCSV(self.geneticPopTwo.convergenceStats)
        self.writeRowToCSV(["POP2 Mean Fitness"])
        self.writeRowToCSV(self.geneticPopTwo.meanStats)
        self.writeRowToCSV(["POP2 Median Fitness"])
        self.writeRowToCSV(self.geneticPopTwo.medianStats)
        self.writeRowToCSV(["POP2 Std Dev"])
        self.writeRowToCSV(self.geneticPopTwo.stdDevStats)
        self.writeRowToCSV([])

    def createCSV(self) -> None:
        """Creates a CSV file for the output data of the run and writes a header"""
        # Build Output Header
        outputHeader = [["GA vs. GA RESULTS", self.runID],
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
                "Daemon Strength", "GA Best Obj Val"]

    def buildGA1Data(self) -> list:
        """Builds a list containing population 1's hyperparameters for exporting to a CSV"""
        return [self.geneticPopOne.populationSize, self.geneticPopOne.numGenerations, self.geneticPopOne.isOneDimAlphaTable,
                self.geneticPopOne.isOptimizedArcSelections, self.geneticPopOne.terminationMethod,
                self.geneticPopOne.stagnationPeriod, self.geneticPopOne.initializationStrategy,
                self.geneticPopOne.initializationDistribution, self.geneticPopOne.initializationParams[0],
                self.geneticPopOne.initializationParams[1], self.geneticPopOne.selectionMethod,
                self.geneticPopOne.tournamentSize, self.geneticPopOne.crossoverMethod, self.geneticPopOne.crossoverRate,
                self.geneticPopOne.crossoverAttemptsPerGeneration, self.geneticPopOne.replacementStrategy,
                self.geneticPopOne.mutationMethod, self.geneticPopOne.mutationRate, self.geneticPopOne.perArcEdgeMutationRate,
                self.geneticPopOne.isDaemonUsed, self.geneticPopOne.annealingConstant, self.geneticPopOne.daemonStrategy,
                self.geneticPopOne.daemonStrength, self.gaSolutionOne.trueCost]

    def buildGA2Data(self) -> list:
        """Builds a list containing population 2's hyperparameters for exporting to a CSV"""
        return [self.geneticPopTwo.populationSize, self.geneticPopTwo.numGenerations, self.geneticPopTwo.isOneDimAlphaTable,
                self.geneticPopTwo.isOptimizedArcSelections, self.geneticPopTwo.terminationMethod,
                self.geneticPopTwo.stagnationPeriod, self.geneticPopTwo.initializationStrategy,
                self.geneticPopTwo.initializationDistribution, self.geneticPopTwo.initializationParams[0],
                self.geneticPopTwo.initializationParams[1], self.geneticPopTwo.selectionMethod,
                self.geneticPopTwo.tournamentSize, self.geneticPopTwo.crossoverMethod, self.geneticPopTwo.crossoverRate,
                self.geneticPopTwo.crossoverAttemptsPerGeneration, self.geneticPopTwo.replacementStrategy,
                self.geneticPopTwo.mutationMethod, self.geneticPopTwo.mutationRate, self.geneticPopTwo.perArcEdgeMutationRate,
                self.geneticPopTwo.isDaemonUsed, self.geneticPopTwo.annealingConstant, self.geneticPopTwo.daemonStrategy,
                self.geneticPopTwo.daemonStrength, self.gaSolutionTwo.trueCost]
