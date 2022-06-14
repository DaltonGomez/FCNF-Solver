import csv
import math
import os
import random
from datetime import datetime

from Utilities.old_code.OLDGraphMaker import GraphMaker
from src.AlphaGenetic.Population import Population
from src.FlowNetwork.CandidateGraph import CandidateGraph
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX


class AlphaGeneticTuning:
    """Class that defines a Tuning Experiment object, used for finding the optimal hyperparameters of the Alpha GA"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, numTrials=5, numGraphs=10, nodeSizeRange=(25, 400), srcSinkSet=(1, 5, 10),
                 arcCostLookupTable=None, srcSinkCapacityRange=(100, 200), srcSinkChargeRange=(10, 25),
                 targetAsPercentTotalDemand=0.50):
        """Constructor of a Tuning Experiment instance"""
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

        # Tuning Output CSV
        now = datetime.now()
        uniqueID = now.strftime("%d_%m_%Y_%H_%M")
        self.fileName = "GA_Tuning_" + uniqueID
        self.createCSV()

        # Build Networks
        self.networkList = self.generateRandomNetworks()

        """
        # ALL HYPERPARAMETERS
        self.populationSizeSet = (10, 50, 100)
        self.numGenerationsSet = (10, 50, 100)
        self.initDistributions = (
            ["uniform", [0.0, 1.0]], ["uniform", [0.0, 10.0]], ["gaussian", [1.0, 0.3]], ["gaussian", [10.0, 1.0]])
        self.terminationMethods = ("setGenerations", "stagnationPeriod")
        self.stagnationPeriodSet = (5, 10, 20)
        # Individual Selection HPs
        self.selectionMethodsAndTournySize = (["roulette", 3], ["random", 3], ["tournament", 3], ["tournament", 8])
        # Path Selection HPs
        self.pathSelectionMethodsAndTournySize = (
            ["roulette", 3], ["random", 3], ["top", 3], ["tournament", 3], ["tournament", 8])
        self.pathRankingOrders = ("most", "least")
        self.pathRankingMethods = ("cost", "flow", "density", "length")
        self.pathSelectionSizeSet = (1, 5, 10)
        # Crossover HPs
        self.crossoverMethods = ("onePoint", "twoPoint", "pathBased")
        self.replacementStrategies = ("replaceWeakestTwo", "replaceParents")
        self.crossoverRateSet = (1.0, 0.75, 0.50)
        self.crossoverAttemptsPerGenerationSet = (1, 3, 5)
        # Mutation HPs
        self.mutationMethods = ("randomSingleArc", "randomSingleEdge", "randomTotal", "pathBasedRandom", "pathBasedNudge")
        self.mutationRateSet = (0.05, 0.10, 0.25, 0.50, 0.75)
        self.nudgeParams = ([0.0, 1.0], [0.0, 0.1], [0.0, 10.0])
        """

        # HYPERPARAMETERS TO TUNE
        self.populationSizeSet = 30
        self.numGenerationsSet = 20
        self.initDistributions = (["uniform", [0.0, 1.0]], ["gaussian", [1.0, 0.3]])
        # Individual Selection HPs
        self.selectionMethodsAndTournySize = (["tournament", 3], ["tournament", 5])
        # Path Selection HPs
        self.pathSelectionMethodsAndTournySize = ("tournament", 8)
        self.pathRankingOrders = ("most", "least")
        self.pathRankingMethods = ("cost", "flow", "density")
        self.pathSelectionSizeSet = (1, 3, 6)
        # Crossover HPs
        self.crossoverMethods = ("onePoint", "twoPoint")
        # self.crossoverMethods = ("onePoint", "twoPoint", "pathBased")  # TODO - Uncomment when pathing fixed
        self.replacementStrategies = ("replaceWeakestTwo", "replaceParents")
        # Mutation HPs
        self.mutationMethods = (["randomSingleEdge", [0.0, 1.0]], ["randomTotal", [0.0, 1.0]])
        # TODO - Uncomment when pathing fixed
        # self.mutationMethods = (["randomSingleEdge", [0.0, 1.0]], ["pathBasedRandom", [0.0, 1.0]], ["pathBasedNudge", [0.0, 1.0]], ["pathBasedNudge", [0.0, 0.1]])
        self.mutationRateSet = (0.01, 0.10, 0.25, 0.50)

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

    def runTuning(self) -> None:
        """Runs a grid search hyperparameter tuning experiment"""
        print("Starting tuning experiment...")
        for networkName in self.networkList:
            print("Solving " + networkName + "...")
            # Initialize Output Row
            networkDataHeaders = ["Node Size", "Src/Sink Size", "Parallel Edges Size", "Target Flow", "OPTIMAL"]
            self.writeRowToCSV(networkDataHeaders)
            networkDataRow = []
            # Parse Input FlowNetwork Data
            networkData = networkName.split("-")
            numNode = int(networkData[0])
            networkDataRow.append(numNode)
            numSrcSinks = int(networkData[1])
            networkDataRow.append(numSrcSinks)
            parallelEdges = int(networkData[2])
            networkDataRow.append(parallelEdges)
            # Load FlowNetwork
            networkFile = networkName + ".p"
            network = CandidateGraph()
            network = network.loadCandidateGraph(networkFile)
            minTargetFlow = math.floor(self.targetAsPercentTotalDemand * network.totalPossibleDemand)
            networkDataRow.append(minTargetFlow)
            # Find Exact Solution
            exactSolver = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
            exactSolver.findSolution(printDetails=False)
            exactValue = exactSolver.model.solution.get_objective_value()
            networkDataRow.append(exactValue)
            print("Exact solution found...")
            # Write FlowNetwork Data and Build Hyperparameter Header
            self.writeRowToCSV(networkDataRow)
            print(networkDataRow)
            hyperparameterHeader = ["Pop Size", "Generations", "Init Dist", "Init Params", "Selection", "Tourny Size",
                                    "Crossover", "Replacement", "Mutation", "Nudge Params", "Mutate Rate",
                                    "Path Select", "Path Tourny", "Path Order", "Path Rank", "Path Size"]
            for n in range(self.numTrials):
                hyperparameterHeader.append("Trial " + str(n + 1))
            hyperparameterHeader.append("Avg Value")
            hyperparameterHeader.append("Opt Gap")
            self.writeRowToCSV(hyperparameterHeader)
            print(hyperparameterHeader)
            # Begin Grid Search
            # NOTE: Intentionally placing path-based operations last in tuning
            for mutation in self.mutationMethods:
                for crossover in self.crossoverMethods:
                    for init in self.initDistributions:
                        for selection in self.selectionMethodsAndTournySize:
                            for replacement in self.replacementStrategies:
                                for mutateRate in self.mutationRateSet:
                                    # Initialize a new population
                                    pop = Population(network, minTargetFlow)
                                    # Set Hyperparameters
                                    pop.setPopulationHyperparams(populationSize=self.populationSizeSet,
                                                                 terminationMethod="setGenerations",
                                                                 numGenerations=self.numGenerationsSet,
                                                                 stagnationPeriod=5,
                                                                 initializationDistribution=init[0],
                                                                 initializationParams=init[1])
                                    pop.setIndividualSelectionHyperparams(selectionMethod=selection[0],
                                                                          tournamentSize=selection[1])
                                    pop.setCrossoverHyperparams(crossoverMethod=crossover,
                                                                replacementStrategy=replacement,
                                                                crossoverRate=1.0,
                                                                crossoverAttemptsPerGeneration=1)
                                    pop.setMutationHyperparams(mutationMethod=mutation[0],
                                                               mutationRate=mutateRate,
                                                               nudgeParams=mutation[1])
                                    # Set Path-Based Operators Hyperparameters
                                    if crossover == "pathBased" or mutation[0] == "pathBasedRandom" or \
                                            mutation[0] == "pathBasedNudge":
                                        for pathOrder in self.pathRankingOrders:
                                            for pathRank in self.pathRankingMethods:
                                                for pathSize in self.pathSelectionSizeSet:
                                                    # Update output row
                                                    outputRow = [self.populationSizeSet, self.numGenerationsSet,
                                                                 init[0], init[1],
                                                                 selection[0], selection[1],
                                                                 crossover, replacement,
                                                                 mutation[0], mutation[1], mutateRate]
                                                    pop.setPathSelectionHyperparams(
                                                        pathSelectionMethod="tournament",
                                                        pathRankingOrder=pathOrder,
                                                        pathRankingMethod=pathRank,
                                                        pathSelectionSize=pathSize,
                                                        pathTournamentSize=8)
                                                    # Write to output row
                                                    outputRow.append("tournament")
                                                    outputRow.append(8)
                                                    outputRow.append(pathOrder)
                                                    outputRow.append(pathRank)
                                                    outputRow.append(pathSize)
                                                    # Solve population for n trials
                                                    trials = []
                                                    for n in range(self.numTrials):
                                                        output = pop.evolvePopulation()
                                                        outputRow.append(output[0])
                                                        trials.append(output[0])
                                                        pop.resetOutputFields()
                                                    # Calculate average and optimality gap
                                                    average = sum(trials) / len(trials)
                                                    outputRow.append(average)
                                                    optGap = ((average / exactValue) - 1) * 100
                                                    outputRow.append(optGap)
                                                    # Write row to csv and print to console
                                                    self.writeRowToCSV(outputRow)
                                                    print(outputRow)
                                    else:
                                        # Update output row
                                        outputRow = [self.populationSizeSet, self.numGenerationsSet, init[0], init[1],
                                                     selection[0], selection[1], crossover, replacement, mutation[0],
                                                     mutation[1], mutateRate]
                                        pop.setPathSelectionHyperparams()
                                        # Write N/A for path-specific hyperparams as they don't apply
                                        for n in range(5):
                                            outputRow.append("N/A")
                                        # Solve population for n trials
                                        trials = []
                                        for n in range(self.numTrials):
                                            output = pop.evolvePopulation()
                                            outputRow.append(output[0])
                                            trials.append(output[0])
                                            pop.resetOutputFields()
                                        # Calculate average and optimality gap
                                        average = sum(trials) / len(trials)
                                        outputRow.append(average)
                                        optGap = ((average / exactValue) - 1) * 100
                                        outputRow.append(optGap)
                                        # Write row to csv and print to console
                                        self.writeRowToCSV(outputRow)
                                        print(outputRow)
        # Timestamp at completion
        now = datetime.now()
        finishTime = now.strftime("%d_%m_%Y_%H_%M")
        self.writeRowToCSV(["Finish Time", finishTime])
        print("\nTUNING EXPERIMENT COMPLETE!")

    def createCSV(self) -> None:
        """Creates the output csv file and writes the header"""
        # Build Output Header
        outputHeader = [["TUNING RESULTS OUTPUT", self.fileName]]
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
