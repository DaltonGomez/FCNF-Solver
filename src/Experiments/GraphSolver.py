from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from src.AlphaGenetic.Population import Population
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX


class GraphSolver:
    """Class that solves a single graph using the alpha-genetic algorithm and/or the MILP model in CPLEX"""

    def __init__(self, inputGraphName: str, isSolvedWithGeneticAlg=True, isSolvedWithCPLEX=True, isRace=True,
                 isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=False):
        """Constructor of a GraphSolver instance"""
        # Graph solver options
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
        self.graph = self.graph.loadCandidateGraph(self.graphName)
        self.minTargetFlow: float = self.graph.totalPossibleDemand

        # Alpha-GA population attribute & hyperparameters
        self.geneticPop: Population = Population(self.graph, self.minTargetFlow,
                         isOneDimAlphaTable=True, isOptimizedArcSelections=True)
        self.geneticPop.setPopulationHyperparams(populationSize=5,
                                                 numGenerations=5,
                                                 initializationStrategy="perEdge",
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

        # MILP CPLEX attribute
        self.milpCplexSolver: MILPsolverCPLEX = MILPsolverCPLEX(self.graph, self.minTargetFlow,
                                          isOneArcPerEdge=False, logOutput=isOutputtingCPLEX)

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
            gaSolution = self.geneticPop.evolvePopulation(printGenerations=True, drawing=self.isDrawing,
                                                          drawLabels=self.isLabeling, isGraphing=self.isGraphing)
            print("\nGenetic Algorithm Complete!!!\nBest Solution Found = " + str(gaSolution.trueCost))
            # Draw if expected
            if self.isDrawing is True:
                solVis = SolutionVisualizer(gaSolution)
                if self.isLabeling is True:
                    solVis.drawLabeledSolution(leadingText="GA-BEST_")
                else:
                    solVis.drawUnlabeledSolution(leadingText="GA-BEST_")
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
                opt = self.milpCplexSolver.writeSolution()
                optVis = SolutionVisualizer(opt)
                if self.isLabeling is True:
                    optVis.drawLabeledSolution(leadingText="OPT_")
                else:
                    optVis.drawUnlabeledSolution(leadingText="OPT_")
            # Print solution details
            print("\nCPLEX MILP Objective Value: " + str(self.milpCplexSolver.getObjectiveValue()))
            print("CPLEX Runtime (in seconds): " + str(self.milpCplexSolver.getCplexRuntime()))
            print("CPLEX Status " + self.milpCplexSolver.getCplexStatus())
            print("CPLEX Gap: " + str(self.milpCplexSolver.getGap() * 100) + "%")
            print("CPLEX Best Bound: " + str(self.milpCplexSolver.getBestBound()))
        if self.isSolvedWithGeneticAlg is True and self.isSolvedWithCPLEX is True and self.isRace is True and self.isGraphing is True:
            self.plotConvergenceAgainstCPLEX()
        print("\n\nProgram complete... Graph solved!\nTerminating program...\n")

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
        timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        plt.savefig("GAvsCPLEX-" + self.graph.name + "-" + timestamp + ".png")
        plt.close(fig)

    def buildGraphDataRow(self, includeHeader=False) -> list:
        """Builds a list containing the graph data for exporting to a CSV"""
        pass

    def buildPopulationHyperparamsRow(self, includeHeader=False) -> list:
        """Builds a list containing the populations hyperparameters for exporting to a CSV"""
        pass

    def buildGeneticEvolutionRow(self, includeHeader=False) -> list:
        """Builds a list containing the genetic algorithm's evolution for exporting to a CSV"""
        pass

    def buildCPLEXDataRow(self, includeHeader=False) -> list:
        """Builds a list containing the solution details of the CPLEX solver on the MILP formulation"""
        pass

    def concatenateDataRows(self) -> list:
        """Builds a list of all the data requested via keyword arguments"""
        pass

    def appendRowToCSV(self, rowToAppend: list) -> None:
        """Appends a row onto a CSV file"""
        pass

    def writeEntireCSV(self, dataBlockToWrite: list) -> None:
        """Writes an entire block of data (i.e. a list of lists) to a CSV file"""
        pass
