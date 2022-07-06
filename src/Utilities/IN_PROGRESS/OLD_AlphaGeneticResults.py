import csv
import math
import os
import random
from datetime import datetime

from Graph.CandidateGraph import CandidateGraph
from Utilities.IN_PROGRESS.OLD_GraphMaker import GraphMaker
from src.AlphaGenetic.Population import Population
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX


class AlphaGeneticResults:
    """Class that defines an Alpha Genetic Results object, used for experimenting on the tuned AlphaGenetic Population"""
    # TODO - Rename AlphaGeneticDriver and rework class
    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, numGraphs=10, nodeSizeRange=(25, 400), srcSinkSet=(1, 5, 10), arcCostLookupTable=None,
                 srcSinkCapacityRange=(100, 200), srcSinkChargeRange=(10, 25), targetAsPercentTotalDemand=0.50,
                 numTrials=10):
        """Constructor of a Results Experiment instance"""
        # Input Parameters
        if arcCostLookupTable is None:
            arcCostLookupTable = [
                [100, 10, 1]
            ]
        self.numGraphs = numGraphs
        self.nodeSizeRange = nodeSizeRange
        self.srcSinkSet = srcSinkSet
        self.arcCostLookupTable = arcCostLookupTable
        self.srcSinkCapacityRange = srcSinkCapacityRange
        self.srcSinkChargeRange = srcSinkChargeRange
        self.targetAsPercentTotalDemand = targetAsPercentTotalDemand
        self.numTrials = numTrials

        # Experimental Results CSV
        now = datetime.now()
        uniqueID = now.strftime("%d_%m_%Y_%H_%M")
        self.fileName = "GA_Results_" + uniqueID
        self.createCSV()

        # Build Networks
        self.networkList = self.generateRandomNetworks()

    def generateRandomNetworks(self) -> list:
        """Creates and saves n new networks based off the input parameters"""
        random.seed()
        networkList = []
        # Automatically generate n input networks
        for n in range(self.numGraphs):
            # Uniformly sample number of nodes
            numNodes = random.randint(self.nodeSizeRange[0], self.nodeSizeRange[1])
            numSrcSinks = random.sample(self.srcSinkSet, 1)[0]
            # Keep sampling until there is enough nodes to support the sources and sinks
            while numSrcSinks > math.floor(numNodes / 2):
                numNodes = random.randint(self.nodeSizeRange[0], self.nodeSizeRange[1])
                numSrcSinks = random.randint(self.srcSinkSet[0], self.srcSinkSet[1])
            # Build and save the network
            networkName = str(numNodes) + "-" + str(numSrcSinks) + "-" + str(len(self.arcCostLookupTable)) + "-" + str(
                n)
            graphMaker = GraphMaker(networkName, numNodes, numSrcSinks, numSrcSinks)
            graphMaker.setArcCostLookupTable(arcCostLookupTable=self.arcCostLookupTable)
            graphMaker.setSourceSinkGeneralizations(isCapacitated=True, isCharged=False,
                                                    srcCapRange=(2, 10), sinkCapRange=(5, 20))
            generatedNetwork = graphMaker.generateNetwork()
            generatedNetwork.saveCandidateGraph()
            networkList.append(networkName)
        return networkList

    def runExperiment(self) -> None:
        """Solves each network optimally, 10x with naive hill climbing, 10x with alpha-genetic, and saves the results"""
        for networkName in self.networkList:
            print("Solving " + networkName + "...")
            # Initialize Output Row
            outputRow = []
            # Parse Input FlowNetwork Data
            networkData = networkName.split("-")
            numNode = int(networkData[0])
            outputRow.append(numNode)
            numSrcSinks = int(networkData[1])
            outputRow.append(numSrcSinks)
            parallelEdges = int(networkData[2])
            outputRow.append(parallelEdges)
            # Load FlowNetwork
            networkFile = networkName + ".p"
            network = CandidateGraph()
            network = network.loadCandidateGraph(networkFile)
            minTargetFlow = math.floor(self.targetAsPercentTotalDemand * network.totalPossibleDemand)
            outputRow.append(minTargetFlow)
            # Find Exact Solution
            exactSolver = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
            exactSolver.findSolution(printDetails=False)
            exactValue = exactSolver.model.solution.get_objective_value()
            outputRow.append(exactValue)
            print("Exact solution found...")
            # Run Hill Climb Trials
            hillClimbTrials = []
            for hillClimbTrial in range(self.numTrials):
                hillClimber = Population(network, minTargetFlow)
                hillClimbOutput = hillClimber.solveWithNaiveHillClimb(printGenerations=False, drawing=False)
                outputRow.append(hillClimbOutput[0])
                hillClimbTrials.append(hillClimbOutput[0])
                print("Hill Climb trial " + str(hillClimbTrial + 1) + " solved...")
            # Find Hill Climb Average & Optimality Gap
            hillClimbAverage = sum(hillClimbTrials) / len(hillClimbTrials)
            outputRow.append(hillClimbAverage)
            hillClimbGap = ((hillClimbAverage / exactValue) - 1) * 100
            outputRow.append(hillClimbGap)
            # Run Alpha Genetic Trials
            geneticTrials = []
            for geneticTrial in range(self.numTrials):
                population = Population(network, minTargetFlow)
                geneticOutput = population.evolvePopulation(printGenerations=False, drawing=False)
                outputRow.append(geneticOutput[0])
                geneticTrials.append(geneticOutput[0])
                print("Alpha Genetic trial " + str(geneticTrial + 1) + " solved...")
            # Find Alpha Genetic Average & Optimality Gap
            geneticAverage = sum(geneticTrials) / len(geneticTrials)
            outputRow.append(geneticAverage)
            geneticGap = ((geneticAverage / exactValue) - 1) * 100
            outputRow.append(geneticGap)
            # Append CSV with Output Row
            self.writeRowToCSV(outputRow)
        # Timestamp at completion
        now = datetime.now()
        finishTime = now.strftime("%d_%m_%Y_%H_%M")
        self.writeRowToCSV(["Finish Time", finishTime])
        print("\nRESULTS EXPERIMENT COMPLETE!!!")

    def createCSV(self) -> None:
        """Creates the output csv file and writes the header"""
        # Build Output Header
        outputHeader = [["EXPERIMENTAL RESULTS OUTPUT", self.fileName],
                        ["Node Size", "Src/Sink Size", "Parallel Edges Size", "Target Flow", "OPTIMAL"]]
        for n in range(self.numTrials):
            outputHeader[1].append("HC " + str(n + 1))
        outputHeader[1].append("Avg HC")
        outputHeader[1].append("HC Opt Gap")
        for m in range(self.numTrials):
            outputHeader[1].append("GA " + str(m + 1))
        outputHeader[1].append("Avg GA")
        outputHeader[1].append("GA Opt Gap")
        # Create CSV File
        currDir = os.getcwd()
        csvName = self.fileName + ".csv"
        catPath = os.path.join(currDir, "data/results/alphaGA", csvName)
        csvFile = open(catPath, "w+", newline="")
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(outputHeader)

    def writeRowToCSV(self, outputRow: list) -> None:
        """Appends the most recent data onto a .csv file"""
        currDir = os.getcwd()
        csvName = self.fileName + ".csv"
        catPath = os.path.join(currDir, "data/results/alphaGA", csvName)
        csvFile = open(catPath, "a", newline="")
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerows([outputRow])
