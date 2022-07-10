import csv
import os
from datetime import datetime

import numpy as np

from Graph.CandidateGraph import CandidateGraph
from Utilities.TABLED_CODE.AntColony.Colony import Colony
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX
from src.Solvers.RelaxedLPSolverPDLP import RelaxedLPSolverPDLP


class AntColonyResults:
    """Class that defines a Results Experiment object, used for comparing the tuned AntColony to the optimal value"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, networkList: list, numAnts: int, numEpisodes: int):
        """Constructor of a Results Experiment instance"""
        # Input Attributes
        self.networkList = networkList
        self.numAnts = numAnts
        self.numEpisodes = numEpisodes

        # Experimental Results
        self.outputBlock = []
        self.buildOutputHeader()
        now = datetime.now()
        uniqueID = now.strftime("%d_%m_%Y_%H_%M")
        self.fileName = "Results_" + uniqueID

    def runExperiment(self) -> None:
        """Solves each network optimally, 10x with the ant colony, and saves the results"""
        for networkName in self.networkList:
            print("Solving " + networkName + "...")
            # Initialize output row
            outputRow = []
            # Get network data and build output row input cells
            networkData = networkName.split("-")
            numNode = networkData[0]
            outputRow.append(numNode)
            parallelEdges = networkData[1]
            outputRow.append(parallelEdges)
            numSrcSinks = networkData[2]
            outputRow.append(numSrcSinks)
            minTargetFlow = int(numSrcSinks) * 100
            outputRow.append(minTargetFlow)
            # Load network
            networkFile = networkName + ".p"
            network = CandidateGraph()
            network = network.loadCandidateGraph(networkFile)
            # Find exact solution
            exactSolver = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
            exactSolver.buildModel()
            exactSolver.solveModel()
            exactValue = exactSolver.model.solution.get_objective_value()
            outputRow.append(exactValue)
            print("Exact solution found...")
            # Find relaxed solution
            relaxedSolver = RelaxedLPSolverPDLP(network, minTargetFlow)
            alphaValues = np.full((network.numEdges, network.numArcsPerEdge), 1.0)
            relaxedSolver.updateObjectiveFunction(alphaValues)
            relaxedSolver.solveModel()
            relaxedSolver.writeSolution()
            outputRow.append(relaxedSolver.trueCost)
            print("Relaxed solution found...")
            # Run AntColony trials
            acoTrials = []
            for trial in range(10):
                aco = Colony(network, minTargetFlow, self.numAnts, self.numEpisodes)
                aco.solveNetwork(drawing=False)
                outputRow.append(aco.bestKnownCost)
                acoTrials.append(aco.bestKnownCost)
                print("AntColony trial " + str(trial) + " solved...")
            # Find AntColony average
            acoAverage = sum(acoTrials) / len(acoTrials)
            outputRow.append(acoAverage)
            # Compute Optimality Gap
            optimalityGap = ((acoAverage / exactValue) - 1) * 100
            outputRow.append(optimalityGap)
            self.writeLineToTxtEnd(outputRow)
            self.outputBlock.append(outputRow)
        self.writeOutputBlock()
        print("\nRESULTS EXPERIMENT COMPLETE!")

    def buildOutputHeader(self) -> None:
        """Builds the header of the output block"""
        now = datetime.now()
        timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
        self.outputBlock.append(["EXPERIMENTAL RESULTS OUTPUT", timestamp])
        self.outputBlock.append(
            ["Num. Nodes", "Num. Parallel Edges", "Num. Src/Sinks", "Min. Target Flow", "Optimal MILP Value",
             "Relaxed LP Value", "AntColony 1", "AntColony 2", "AntColony 3", "AntColony 4", "AntColony 5",
             "AntColony 6", "AntColony 7", "AntColony 8", "AntColony 9",
             "AntColony 10", "Avg. AntColony Value", "Optimality Gap"])

    def writeOutputBlock(self) -> None:
        """Writes the output block to a csv file"""
        currDir = os.getcwd()
        csvName = self.fileName + ".csv"
        catPath = os.path.join(currDir, "data/results/antColony", csvName)
        file = open(catPath, "w+", newline="")
        with file:
            write = csv.writer(file)
            write.writerows(self.outputBlock)

    def writeLineToTxtEnd(self, outputRow: list) -> None:
        """Appends the most recent data onto a .txt file"""
        currDir = os.getcwd()
        txtName = self.fileName + ".txt"
        catPath = os.path.join(currDir, "data/results/antColony", txtName)
        with open(catPath, "a") as file_object:
            file_object.write(str(outputRow) + "\n")
