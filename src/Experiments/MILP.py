import csv
import os
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt

from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph
from src.Graph.GraphVisualizer import GraphVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX


class MILP:
    """Class that solves a single graph the MILP model in CPLEX"""

    def __init__(self, inputGraphName: str, isTimeConstrained=False, timeLimit=-1.0, isOneArcPerEdge=True,
                 isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=True, isSolutionSaved=True):
        """Constructor of a GAvsMILP instance"""
        # Graph solver options
        self.runID = "MILP--" + inputGraphName + "--" + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.isTimeConstrained: bool = isTimeConstrained
        self.timeLimit: float = timeLimit
        self.isOneArcPerEdge: bool = isOneArcPerEdge
        self.isDrawing: bool = isDrawing
        self.isLabeling: bool = isLabeling
        self.isGraphing: bool = isGraphing
        if self.isGraphing is True:
            matplotlib.use("agg")  # Simpler MatPlotLib backend for rendering high number of PNGs per run
        self.isOutputtingCPLEX: bool = isOutputtingCPLEX
        self.isSolutionSaved: bool = isSolutionSaved

        # Input graph attributes
        self.graphName: str = inputGraphName
        self.graph: CandidateGraph = CandidateGraph()
        self.graph = self.graph.loadCandidateGraph(self.graphName + ".p")
        self.minTargetFlow: float = self.graph.totalPossibleDemand
        # Visualize input
        if self.isDrawing is True:
            graphVis = GraphVisualizer(self.graph)
            if self.isLabeling is True:
                graphVis.drawLabeledGraph()
                # graphVis.drawBidirectionalGraphWithSmoothedLabeledEdges()
            else:
                graphVis.drawUnlabeledGraph()
                # graphVis.drawBidirectionalGraph()

        # MILP CPLEX attribute
        self.milpCplexSolver: MILPsolverCPLEX = MILPsolverCPLEX(self.graph, self.minTargetFlow,
                                          isOneArcPerEdge=self.isOneArcPerEdge, logOutput=isOutputtingCPLEX)
        if self.isTimeConstrained is True:
            if self.timeLimit == -1.0:
                self.milpCplexSolver.setTimeLimit(float(input("Please provide time limit for CPLEX:\n-> ")))
            else:
                self.milpCplexSolver.setTimeLimit(self.timeLimit)
        self.milpSolution = None
        self.milpRuntimeInSeconds: float = -1

    def solveGraph(self) -> None:
        """Solves the graph with the MILP formulation in CPLEX"""
        # Solve the MILP formulation in CPLEX
        print("\n============================================================================")
        print("Solving the " + self.graphName + " graph with a MILP formulation in CPLEX...\n")
        # Call CPLEX to solve MILP
        self.milpSolution = self.milpCplexSolver.findSolution(printDetails=False)
        print("\nCPLEX MILP Solver Complete!!!\nBest Solution Found = " + str(self.milpCplexSolver.getObjectiveValue()))
        # Draw if expected
        if self.isDrawing is True:
            milpVis = SolutionVisualizer(self.milpSolution)
            if self.isLabeling is True:
                milpVis.drawLabeledSolution(leadingText="MILP_")
            else:
                milpVis.drawUnlabeledSolution(leadingText="MILP_")
        # Print solution details
        print("\nMILP Objective Value: " + str(self.milpCplexSolver.getObjectiveValue()))
        print("MILP Runtime (in seconds): " + str(self.milpCplexSolver.getCplexRuntime()))
        print("MILP Status " + self.milpCplexSolver.getCplexStatus())
        print("MILP Gap: " + str(self.milpCplexSolver.getGap() * 100) + "%")
        print("MILP Best Bound: " + str(self.milpCplexSolver.getBestBound()))
        if self.isGraphing is True:
            self.plotCplexRuntimeStats()
        self.saveOutputAsCSV()
        if self.isSolutionSaved is True:
            self.milpSolution.saveSolution()
        print("\nRun complete!\n")

    def plotCplexRuntimeStats(self) -> None:
        """Plots the MILP's runtime states collected by a CPLEX listener"""
        # Get plt figure
        fig = plt.figure()
        ax = fig.add_subplot()
        # Plot all data
        ax.plot(self.milpCplexSolver.runtimeTimestamps, self.milpCplexSolver.runtimeObjectiveValues, label="MILP Obj Val", color="y", linestyle=":")
        ax.plot(self.milpCplexSolver.runtimeTimestamps, self.milpCplexSolver.runtimeBestBounds, label="MILP Bound", color="r", linestyle=":")
        # Add graph elements
        ax.set_title("MILP Runtime Stats")
        ax.legend(loc=1)
        ax.set_ylim(ymin=0, ymax=max(self.milpCplexSolver.runtimeObjectiveValues)/1.5)
        ax.set_ylabel("Obj. Value")
        ax.set_xlabel("Runtime (in sec)")
        # Save timestamped plot
        plt.savefig(self.runID + ".png")
        plt.close(fig)

    def saveOutputAsCSV(self) -> None:
        """Writes the output data to disc as a CSV file"""
        print("\nWriting output to disc as '" + self.runID + ".csv'...")
        self.createCSV()
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
        outputHeader = [["MILP RESULTS", self.runID],
                        [],
                        ["INPUT GRAPH DATA"],
                        ["Graph Name", "Num Nodes", "Num Sources", "Num Sinks", "Num Edges",
                         "Num Arc Caps", "Target Flow", "is Src/Sink Capped?", "is Src/Sink Charged?",
                         "is One Arc Per Edge?"],
                        [self.graphName, self.graph.numTotalNodes, self.graph.numSources, self.graph.numSinks,
                         self.graph.numEdges, self.graph.numArcsPerEdge, self.minTargetFlow,
                         self.graph.isSourceSinkCapacitated, self.graph.isSourceSinkCharged, self.isOneArcPerEdge],
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
    def buildMILPHeaderRow() -> list:
        """Builds a list containing the solution detail headers of the MILP formulation in CPLEX"""
        return ["MILP Obj Val", "MILP Runtime (sec)", "Time Limit", "Status", "Status Code", "Best Bound",
                "MILP Gap"]

    def buildMILPDataRow(self) -> list:
        """Builds a list containing the solution details of the MILP formulation in CPLEX"""
        return [self.milpCplexSolver.getObjectiveValue(), self.milpCplexSolver.getCplexRuntime(),
                self.milpCplexSolver.getTimeLimit(), self.milpCplexSolver.getCplexStatus(),
                self.milpCplexSolver.getCplexStatusCode(), self.milpCplexSolver.getBestBound(),
                self.milpCplexSolver.getGap()]
