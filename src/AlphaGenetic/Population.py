import random
import sys
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from src.AlphaGenetic.AlphaSolverCPLEX import AlphaSolverCPLEX
from src.AlphaGenetic.Individual import Individual
from src.FlowNetwork.FlowNetworkSolution import FlowNetworkSolution
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph


class Population:
    """Class that manages a population of alpha-relaxed individuals and handles their evolution with GA operators"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, graph: CandidateGraph, minTargetFlow: float, populationSize=10, numGenerations=10,
                 isOneDimAlphaTable=True, isOptimizedArcSelections=True, isPenalizedObjective=False):
        """Constructor of a Population instance"""
        # Input Attributes
        self.graph: CandidateGraph = graph  # Input candidate graph instance to be solved
        self.minTargetFlow: float = minTargetFlow  # Input target flow to be realized in output solution
        # Population & Solver Instance Objects
        self.population: List[Individual] = []  # List of Individual objects
        self.currentGeneration: int = 0  # Tracks generation number during evolution
        self.solver: AlphaSolverCPLEX = AlphaSolverCPLEX(self.graph, self.minTargetFlow,
                                                       isOneDimAlphaTable=isOneDimAlphaTable,
                                                       isOptimizedArcSelections=isOptimizedArcSelections,
                                                       isPenalizedObjective=isPenalizedObjective)  # Solver object, which pre-builds variables and constraints once on initialization
        self.generationTimestamps: List[float] = []  # List of timestamps, in seconds after evolution start, for each generation
        self.isTerminated: bool = False  # Boolean indicating if the termination criteria has been reached
        self.bestKnownCost: float = sys.maxsize  # Holds the true cost of the best solution discovered during the evolution
        self.bestKnownAlphas: ndarray = np.array(0, dtype='f')  # Holds the alpha values of the best solution discovered during the evolution
        self.bestKnownSolution = None  # Holds the best (i.e. lowest cost) solution discovered during the evolution

        # =======================
        # GA HYPERPARAMETERS
        # -----------------------
        # Global hyperparameters
        self.populationSize: int = populationSize  # Defines the number of individuals in the population
        self.numGenerations: int = numGenerations  # Defines the number of iterations the evolution loop will run for
        self.isOneDimAlphaTable: bool = isOneDimAlphaTable  # Boolean indicating if the alpha table is only one dimensional (i.e. only one arc per edge)
        self.isOptimizedArcSelections: bool = isOptimizedArcSelections  # Boolean indicating post-processing individuals to best fit the assigned flow to a capacity
        # Initialization HPs
        self.terminationMethod: str = "setGenerations"  # :param : "setGenerations", "stagnationPeriod"
        self.stagnationPeriod: int = 5
        self.consecutiveStagnantGenerations: int = 0
        self.initializationStrategy: str = "perEdge"  # :param : "perArc", "perEdge", "reciprocalCap"
        self.initializationDistribution: str = "gaussian"  # :param : "uniform", "gaussian", "digital"
        self.initializationParams: List[float] = [500.0, 100.0]  # :param: range if uniform distribution, mu and sigma if Gaussian, low and high value if digital
        # Individual Selection HPs
        self.selectionMethod: str = "tournament"  # :param : "tournament", "roulette", "random"
        self.tournamentSize: int = 3
        # Crossover HPs
        self.crossoverMethod: str = "onePoint"  # :param : "onePoint", "twoPoint"
        self.crossoverRate: float = 1.0
        self.crossoverAttemptsPerGeneration: int = 1
        self.replacementStrategy: str = "replaceWeakestTwo"  # : param : "replaceWeakestTwo", "replaceParents", "replaceRandomTwo"
        # Mutation HPs
        self.mutationMethod: str = "randomPerEdge"  # :param : "randomSingleArc", "randomSingleEdge", "randomPerArc", "randomPerEdge", "randomTotal"
        self.mutationRate: float = 0.05
        self.perArcEdgeMutationRate: float = 0.20
        # Daemon HPs
        self.isDaemonUsed: bool = True
        self.annealingConstant: float = 0.5
        self.daemonStrategy: str = "globalMean"  # :param : "globalBinary", "globalMean", "globalMedian", "personalMean", "personalMedian"
        self.daemonStrength: float = 1.0

        # =======================
        # EVOLUTION STATISTICS
        # -----------------------
        self.convergenceStats: List = []  # Logs the objective value of the best individual in the population at each generation
        self.meanStats: List = []  # Logs the mean objective value of the population at each generation
        self.medianStats: List = []  # Logs the median objective value of the population at each generation
        self.stdDevStats: List = []  # Logs the standard deviation of the population at each generation

    # ====================================================
    # ============== HYPERPARAMETER SETTERS ==============
    # ====================================================
    def setPopulationHyperparams(self, populationSize=10, numGenerations=10,
                                 terminationMethod="setGenerations", stagnationPeriod=5) -> None:
        """Sets the GA class field that dictates the range when randomly initializing/updating alpha values \n
        :param int populationSize: Number of individuals in the GA population
        :param int numGenerations: Number of iterations the population evolves for
        :param str terminationMethod: One of following: {"setGenerations", "stagnationPeriod"}
        :param int stagnationPeriod: Number of stagnant consecutive generations needed for termination
        """
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.terminationMethod = terminationMethod
        self.stagnationPeriod = stagnationPeriod

    def setInitializationHyperparams(self, initializationStrategy="perEdge", initializationDistribution="gaussian",
                                 initializationParams=(500.0, 100.0)) -> None:
        """Sets the GA attributes that dictate the initialization/updating of alpha values \n
        :param str initializationStrategy: One of following: {"perEdge", "perArc", "reciprocalCap"}
        :param str initializationDistribution: One of following: {"uniform", "gaussian", "digital"}
        :param list initializationParams: Lower and upper bounds if uniform distribution; mu and sigma if Gaussian; low and high value if digital
        """
        self.initializationStrategy = initializationStrategy
        self.initializationDistribution = initializationDistribution
        self.initializationParams = initializationParams

    def setIndividualSelectionHyperparams(self, selectionMethod="tournament", tournamentSize=3) -> None:
        """Sets the GA attributes that dictate how the selection of individuals is carried out \n
        :param str selectionMethod: One of following: {"tournament", "roulette", "random"}
        :param int tournamentSize: Size of tournament subset used if selectionMethod = "tournament"
        """
        self.selectionMethod = selectionMethod
        self.tournamentSize = tournamentSize

    def setCrossoverHyperparams(self, crossoverMethod="onePoint", crossoverRate=1.0,
                                crossoverAttemptsPerGeneration=1, replacementStrategy="replaceWeakestTwo") -> None:
        """Sets the GA attributes that dictate how the crossover of individuals is carried out \n
        :param str crossoverMethod: One of following: {"onePoint", "twoPoint"}
        :param float crossoverRate: Probability in [0,1] that a crossover occurs
        :param int crossoverAttemptsPerGeneration: Number of attempted crossovers per generation
        :param str replacementStrategy: One of following: {"replaceWeakestTwo", "replaceParents", "replaceRandomTwo"}
        """
        self.crossoverMethod = crossoverMethod
        self.crossoverRate = crossoverRate
        self.crossoverAttemptsPerGeneration = crossoverAttemptsPerGeneration
        self.replacementStrategy = replacementStrategy

    def setMutationHyperparams(self, mutationMethod="randomPerEdge", mutationRate=0.05, perArcEdgeMutationRate=0.20) -> None:
        """Sets the GA attributes that dictate how the mutation of individuals is carried out \n
        :param str mutationMethod: One of following: {"randomSingleArc", "randomSingleEdge", "randomPerArc", "randomPerEdge", "randomTotal"}
        :param float mutationRate: Probability in [0,1] that an individual mutates
        :param float perArcEdgeMutationRate: Probability in [0,1] that an edge/arc mutates given that an individual mutates
        """
        self.mutationMethod = mutationMethod
        self.mutationRate = mutationRate
        self.perArcEdgeMutationRate = perArcEdgeMutationRate

    def setDaemonHyperparams(self, isDaemonUsed=True, annealingConstant=0.5, daemonStrategy="globalMean",
                             daemonStrength=1.0) -> None:
        """Sets the GA attributes that determine the behavior of the annealed daemon update \n
        :param bool isDaemonUsed: Boolean indicating if a daemon update is attempted
        :param float annealingConstant: Constant k in the annealing schedule t = k*gen/(gen + maxGen) - 2.0 = 1.0 proportion on the final generation
        :param str daemonStrategy: One of following: {"globalBinary", "globalMean", "globalMedian", "personalMean", "personalMedian"}
        :param float daemonStrength: Constant that determines how great of an impact the daemon updates have
        """
        self.isDaemonUsed = isDaemonUsed
        self.annealingConstant = annealingConstant
        self.daemonStrategy = daemonStrategy
        self.daemonStrength = daemonStrength

    # ====================================================
    # ============== EVOLUTION LOOP/METHODS ==============
    # ====================================================
    def evolvePopulation(self, printGenerations=True, drawing=True, drawLabels=False,
                         isGraphing=True, runID="") -> FlowNetworkSolution:
        """Evolves the population for a specified number of generations"""
        # Initialize population and solve
        startTime = datetime.now()
        self.initializePopulation()
        self.solvePopulation()
        if isGraphing is True:
            self.computeEvolutionStatistics()
        self.currentGeneration = 1
        self.logGenerationTimestamp(startTime)
        # Evolve population
        while self.isTerminated is not True:
            print("Starting Generation " + str(self.currentGeneration) + "...")
            # Perform operators and solve
            self.selectAndCrossover()
            self.doMutations()
            # Apply a daemon update if used
            if self.isDaemonUsed is True and self.currentGeneration != 1:
                self.enactDaemon()
            self.solvePopulation()
            # Update current best individual and evaluate termination
            bestIndividual = self.getMostFitIndividual()
            self.evaluateTermination(self.currentGeneration, bestIndividual.trueCost)
            # Compute statistics
            if isGraphing is True:
                self.computeEvolutionStatistics()
            # Visualize & print
            if printGenerations is True:
                print("Generation = " + str(self.currentGeneration) + "\t\tBest Individual = " + str(
                    bestIndividual.id) + "\t\tFitness = " + str(round(bestIndividual.trueCost, 2)) + "\n")
            if drawing is True:
                self.visualizeBestIndividual(labels=drawLabels, leadingText="GA_Gen" + str(self.currentGeneration) + "_")
            self.currentGeneration += 1
            self.logGenerationTimestamp(startTime)
        # Plot statistics
        if isGraphing is True:
            self.plotEvolutionStatistics(runID=runID)
        # Return best solution
        return self.bestKnownSolution

    def selectAndCrossover(self) -> None:
        """Performs the selection and crossover at each generation"""
        random.seed()
        for n in range(self.crossoverAttemptsPerGeneration):
            if random.random() < self.crossoverRate:
                parents = self.selectIndividuals()
                self.crossover(parents[0], parents[1])

    def doMutations(self) -> None:
        """Performs mutations on individuals given a mutation probability"""
        random.seed()
        for individualID in range(self.populationSize):
            individual = self.population[individualID]
            if individual.isSolved is False:
                continue
            if random.random() < self.mutationRate:
                self.mutate(individualID)

    def enactDaemon(self) -> None:
        """Applies a daemon update to the population"""
        random.seed()
        annealedDaemonProb = self.getAnnealedDaemonRate()
        for individualID in range(self.populationSize):
            individual = self.population[individualID]
            if individual.isSolved is False:
                continue
            if random.random() < annealedDaemonProb:
                self.applyDaemonUpdate(individualID)

    def evaluateTermination(self, generation: int, newBestCost: float) -> None:
        """Checks for termination using the given method and updates the best known solution"""
        if self.terminationMethod == "setGenerations":
            if newBestCost < self.bestKnownCost:
                self.bestKnownCost = newBestCost
                self.bestKnownAlphas = self.getMostFitIndividual().alphaValues.copy()
                self.bestKnownSolution = self.writeIndividualsSolution(self.getMostFitIndividual())
            if generation >= self.numGenerations:
                self.isTerminated = True
        elif self.terminationMethod == "stagnationPeriod":
            if newBestCost < self.bestKnownCost:
                self.bestKnownCost = newBestCost
                self.bestKnownAlphas = self.getMostFitIndividual().alphaValues.copy()
                self.bestKnownSolution = self.writeIndividualsSolution(self.getMostFitIndividual())
                self.consecutiveStagnantGenerations = 0
            elif newBestCost >= self.bestKnownCost:
                self.consecutiveStagnantGenerations += 1
                if self.consecutiveStagnantGenerations >= self.stagnationPeriod:
                    self.isTerminated = True
        else:
            print("ERROR - INVALID TERMINATION METHOD!!!")

    # ====================================================
    # ============== INITIALIZATION METHODS ==============
    # ====================================================
    def initializePopulation(self) -> None:
        """Initializes the GA population with random alpha values"""
        for individual in range(self.populationSize):
            thisGenotype = self.getInitialAlphaValues()
            thisIndividual = Individual(individual, self.graph, thisGenotype)
            self.population.append(thisIndividual)

    def getInitialAlphaValues(self) -> ndarray:
        """Returns a randomly initialized array of alpha values (i.e. the genotype)"""
        tempAlphaValues = []
        for edge in range(self.graph.numEdges):
            tempEdge = []
            if self.initializationStrategy == "perArc":
                for cap in range(self.graph.numArcsPerEdge):
                    thisArcsAlphaValue = self.getAlphaValue()
                    tempEdge.append(thisArcsAlphaValue)
            elif self.initializationStrategy == "perEdge":
                thisEdgesAlphaValue = self.getAlphaValue()
                for cap in range(self.graph.numArcsPerEdge):
                    tempEdge.append(thisEdgesAlphaValue)
            elif self.initializationStrategy == "reciprocalCap":
                # ASK - Does the reciprocalCap initialization strategy have any merit? I don't think so!
                # TODO - Remove the reciprocalCap initialization strategy
                for cap in range(self.graph.numArcsPerEdge):
                    thisArcsAlphaValue = self.getReciprocalOfMinCap(cap)
                    tempEdge.append(thisArcsAlphaValue)
            else:
                print("ERROR - INVALID INITIALIZATION STRATEGY!!!")
            tempAlphaValues.append(tempEdge)
        initialGenotype = np.array(tempAlphaValues, dtype='f')
        return initialGenotype

    def getAlphaValue(self) -> float:
        """Returns a single alpha value for population initialization"""
        random.seed()
        randomGene = 1.0
        if self.initializationDistribution == "uniform":
            randomGene = random.uniform(self.initializationParams[0], self.initializationParams[1])
        elif self.initializationDistribution == "gaussian":
            randomGene = random.gauss(self.initializationParams[0], self.initializationParams[1])
        elif self.initializationDistribution == "digital":
            randomGene = random.choice(self.initializationParams)
        else:
            print("ERROR - INVALID INITIALIZATION DISTRIBUTION!!!")
        if randomGene < 0.0:
            randomGene = 0.0
        return randomGene

    def getReciprocalOfMinCap(self, arcIndex: int) -> float:
        """Returns the reciprocal of the lower bound of the capacity for the arc size"""
        # If the arc is the smallest, return an arbitrarily large number
        if arcIndex == 0:
            return 100
        # Else return the reciprocal of the capacity just below this arc
        else:
            return 1 / (self.graph.possibleArcCapsArray[arcIndex-1] + 0.01)

    # =============================================
    # ============== RANKING METHODS ==============
    # =============================================
    def rankPopulation(self) -> list:
        """Ranks the population OUT-OF-PLACE in ascending order of cost (i.e. Lower cost -> More fit) and returns"""
        sortedPopulation = sorted(self.population, key=lambda x: x.trueCost)
        return sortedPopulation

    def getMostFitIndividual(self) -> Individual:
        """Returns the most fit individual in the population"""
        return self.population[self.getMostFitIndividualID()]

    def getMostFitIndividualID(self) -> int:
        """Returns the most fit individual in the population"""
        sortedPopulation = self.rankPopulation()
        return sortedPopulation[0].id

    def getWeakestTwoIndividualIDs(self) -> tuple:
        """Returns the IDs of the two least fit individuals in the population"""
        sortedPopulation = self.rankPopulation()
        return sortedPopulation[-1].id, sortedPopulation[-2].id

    # ============================================
    # ============== SOLVER METHODS ==============
    # ============================================
    def solvePopulation(self) -> None:
        """Solves all unsolved instances in the entire population"""
        for individualNum, individual in enumerate(self.population):
            if individual.isSolved is False:
                print("Solving individual " + str(individualNum) + "...")
                self.solveIndividual(individualNum)

    def solveIndividual(self, individualNum: int) -> None:
        """Solves a single individual and writes the expressed network to the individual"""
        individual = self.population[individualNum]
        # Overwrite new objective function with new alpha values and solve
        self.solver.updateObjectiveFunction(individual.alphaValues)
        self.solver.solveModel()
        # Write expressed network output data to individual
        individual.isSolved = True
        individual.arcFlows = self.solver.getArcFlowsDict()
        individual.srcFlows = self.solver.getSrcFlowsList()
        individual.sinkFlows = self.solver.getSinkFlowsList()
        if self.solver.isOptimizedArcSelections is True:
            individual.arcFlows = self.solver.optimizeArcSelection(individual.arcFlows)
        individual.trueCost = self.solver.calculateTrueCost()
        individual.fakeCost = self.solver.getObjectiveValue()
        # If no solution was found, hypermutate individual and recursively solve until solution is found
        if individual.trueCost == 0:
            print("ERROR: Individual " + str(individualNum) + " is infeasible! Hypermutating individual...")
            self.hypermutateIndividual(individualNum)
            self.solveIndividual(individualNum)
        # Reset solver
        self.solver.resetSolver()

    def writeIndividualsSolution(self, individual: Individual) -> FlowNetworkSolution:
        """Writes the individual's output to as solution object"""
        solution = individual.writeIndividualAsSolution(self.minTargetFlow, optionalDescription="1D_Alpha_Table = " +
                                                                                        str(self.isOneDimAlphaTable))
        return solution

    def resetOutputFields(self) -> None:
        """Resets the output fields stored in the population"""
        self.isTerminated = False
        self.population = []
        self.solver = AlphaSolverCPLEX(self.graph, self.minTargetFlow,
                                       isOneDimAlphaTable=self.isOneDimAlphaTable,
                                       isOptimizedArcSelections=self.isOptimizedArcSelections)
        self.bestKnownSolution = None
        self.bestKnownCost = sys.maxsize

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def visualizeIndividual(self, individual: Individual, labels=False, leadingText="") -> None:
        """Renders the visualization for a specified individual"""
        solution = self.writeIndividualsSolution(individual)
        visualizer = SolutionVisualizer(solution)
        if labels is True:
            visualizer.drawLabeledSolution(leadingText=leadingText)
        else:
            visualizer.drawUnlabeledSolution(leadingText=leadingText)

    def visualizeBestIndividual(self, labels=False, leadingText="") -> None:
        """Renders the visualization for the most fit individual in the population at any time"""
        bestIndividual = self.getMostFitIndividual()
        self.visualizeIndividual(bestIndividual, labels=labels, leadingText=leadingText)

    def visualizeAllIndividuals(self, labels=False, leadingText="") -> None:
        """Renders the visualization for all individuals in the population at any time"""
        i = 0
        for individual in self.population:
            self.visualizeIndividual(individual, labels=labels, leadingText=leadingText + "_" + str(i))
            i += 1

    # ============================================================
    # ============== INDIVIDUAL SELECTION OPERATORS ==============
    # ============================================================
    def selectIndividuals(self) -> list:
        """Hyper-selection operator that calls specific selection method based on hyperparameters"""
        selectedIndividualIDs = []
        if self.selectionMethod == "tournament":
            selectedIndividualIDs = self.tournamentSelection()
        elif self.selectionMethod == "roulette":
            selectedIndividualIDs = self.rouletteWheelSelection()
        elif self.selectionMethod == "random":
            selectedIndividualIDs = self.randomSelection()
        else:
            print("ERROR - INVALID SELECTION METHOD!!!")
        return selectedIndividualIDs

    def randomSelection(self) -> list:
        """Returns a random subset of individuals in the population (w/o replacement)"""
        random.seed()
        individualsIDs = random.sample(range(len(self.population)), 2)
        return individualsIDs

    def rouletteWheelSelection(self) -> list:
        """Selects individuals probabilistically by their normalized fitness"""
        random.seed()
        sortedPopulation = self.rankPopulation()
        fitnessFromCost = []
        for individual in range(len(sortedPopulation)):
            fitnessFromCost.append(1 / sortedPopulation[individual].trueCost)
        cumulativeFitness = 0
        for individual in range(len(sortedPopulation)):
            cumulativeFitness += fitnessFromCost[individual]
        cumulativeProbabilities = [fitnessFromCost[0] / cumulativeFitness]
        for i in range(1, len(sortedPopulation)):
            cumulativeProbabilities.append(
                (fitnessFromCost[i] / cumulativeFitness) + cumulativeProbabilities[i - 1])
        selectionSet = set()
        while len(selectionSet) < 2:
            rng = random.random()
            for individual in range(len(sortedPopulation)):
                if rng < cumulativeProbabilities[individual]:
                    selectionSet.add(individual)
                    break
        individualIDs = []
        for individual in selectionSet:
            individualIDs.append(sortedPopulation[individual].id)
        return individualIDs

    def tournamentSelection(self) -> list:
        """Selects the best k individuals out of a randomly chosen subset of size n"""
        random.seed()
        # Select a random subset of population
        subset = random.sample(range(len(self.population)), self.tournamentSize)
        # Sort by cost
        tournament = []
        for individual in subset:
            cost = self.population[individual].trueCost
            tournament.append((individual, cost))
        tournament.sort(key=lambda c: c[1], reverse=False)
        individualsIDs = []
        for i in range(2):
            topPick = tournament.pop(0)
            individualsIDs.append(topPick[0])
        return individualsIDs

    # ==============================================
    # ============== MUTATION METHODS ==============
    # ==============================================
    def mutate(self, individualID: int) -> None:
        """Hyper-selection operator that calls specific mutation method based on hyperparameters"""
        print("Performing mutation on individual " + str(individualID) + "...")
        if self.mutationMethod == "randomTotal":
            self.hypermutateIndividual(individualID)
        elif self.mutationMethod == "randomSingleArc":
            self.randomSingleArcMutation(individualID)
        elif self.mutationMethod == "randomSingleEdge":
            self.randomSingleEdgeMutation(individualID)
        elif self.mutationMethod == "randomPerArc":
            self.randomPerArcMutation(individualID)
        elif self.mutationMethod == "randomPerEdge":
            self.randomPerEdgeMutation(individualID)
        else:
            print("ERROR - INVALID MUTATION METHOD!!!")

    def randomSingleArcMutation(self, individualID: int) -> None:
        """Mutates an individual at only one random arc in the chromosome"""
        random.seed()
        mutatedEdge = random.randint(0, self.graph.numEdges - 1)
        mutatedCap = random.randint(0, self.graph.numArcsPerEdge - 1)
        individual = self.population[individualID]
        individual.alphaValues[mutatedEdge][mutatedCap] = self.getAlphaValue()
        individual.resetOutputNetwork()

    def randomSingleEdgeMutation(self, individualID: int) -> None:
        """Mutates an individual at all arcs in a random edge in the chromosome"""
        random.seed()
        mutatedEdge = random.randint(0, self.graph.numEdges - 1)
        individual = self.population[individualID]
        if self.initializationStrategy == "perArc":
            for arcIndex in range(self.graph.numArcsPerEdge):
                individual.alphaValues[mutatedEdge][arcIndex] = self.getAlphaValue()
        elif self.initializationStrategy == "perEdge":
            thisEdgeAlpha = self.getAlphaValue()
            for arcIndex in range(self.graph.numArcsPerEdge):
                individual.alphaValues[mutatedEdge][arcIndex] = thisEdgeAlpha
        individual.resetOutputNetwork()

    def randomPerArcMutation(self, individualID: int) -> None:
        """Iterates over all (edge, arc) pairs and mutates if the perArcEdgeMutation rate rng rolls"""
        random.seed()
        individual = self.population[individualID]
        for edge in range(self.graph.numEdges):
            for arcIndex in range(self.graph.numArcsPerEdge):
                if random.random() < self.perArcEdgeMutationRate:
                    individual.alphaValues[edge][arcIndex] = self.getAlphaValue()
        individual.resetOutputNetwork()

    def randomPerEdgeMutation(self, individualID: int) -> None:
        """Iterates over all edges and mutates at all arcs if the perArcEdgeMutation rate rng rolls"""
        random.seed()
        individual = self.population[individualID]
        for edge in range(self.graph.numEdges):
            if random.random() < self.perArcEdgeMutationRate:
                if self.initializationStrategy == "perArc":
                    for arcIndex in range(self.graph.numArcsPerEdge):
                        individual.alphaValues[edge][arcIndex] = self.getAlphaValue()
                elif self.initializationStrategy == "perEdge":
                    thisEdgeAlpha = self.getAlphaValue()
                    for arcIndex in range(self.graph.numArcsPerEdge):
                        individual.alphaValues[edge][arcIndex] = thisEdgeAlpha
        individual.resetOutputNetwork()

    # =================================================
    # ============== CROSSOVER OPERATORS ==============
    # =================================================
    def crossover(self, parentOneID: int, parentTwoID: int) -> None:
        """Hyper-selection operator that calls specific crossover method based on hyperparameters"""
        print("Performing crossover on individual " + str(parentOneID) + " and individual " + str(parentTwoID) + "...")
        if self.crossoverMethod == "onePoint":
            self.randomOnePointCrossover(parentOneID, parentTwoID)
        elif self.crossoverMethod == "twoPoint":
            self.randomTwoPointCrossover(parentOneID, parentTwoID)
        else:
            print("ERROR - INVALID CROSSOVER METHOD!!!")

    def randomOnePointCrossover(self, parentOneID: int, parentTwoID: int) -> None:
        """Crossover of 2 chromosomes at a single random point\n
        :param int parentOneID: Index of first parent in population
        :param int parentTwoID: Index of second parent in population
        """
        random.seed()
        # Generate crossover point
        crossoverPoint = random.randint(1, self.graph.numEdges - 2)
        # Simplify parent chromosomes naming
        parentOne = self.population[parentOneID].alphaValues
        parentTwo = self.population[parentTwoID].alphaValues
        # Perform crossover
        childOne = np.vstack((parentOne[:crossoverPoint], parentTwo[crossoverPoint:]))
        childTwo = np.vstack((parentTwo[:crossoverPoint], parentOne[crossoverPoint:]))
        # Do replacement with offspring
        self.replaceWithOffspring(parentOneID, parentTwoID, childOne, childTwo)

    def randomTwoPointCrossover(self, parentOneID: int, parentTwoID: int) -> None:
        """Crossover of 2 chromosomes at two random points\n
        :param int parentOneID: Index of first parent in population
        :param int parentTwoID: Index of second parent in population
        """
        random.seed()
        # Generate crossover points
        crossoverOne = random.randint(0, self.graph.numEdges - 3)
        crossoverTwo = random.randint(crossoverOne, self.graph.numEdges - 1)
        # Simplify parent chromosomes naming
        parentOne = self.population[parentOneID].alphaValues
        parentTwo = self.population[parentTwoID].alphaValues
        # Perform crossover
        childOne = np.vstack((parentOne[:crossoverOne], parentTwo[crossoverOne:crossoverTwo], parentOne[crossoverTwo:]))
        childTwo = np.vstack((parentTwo[:crossoverOne], parentOne[crossoverOne:crossoverTwo], parentTwo[crossoverTwo:]))
        # Do replacement with offspring
        self.replaceWithOffspring(parentOneID, parentTwoID, childOne, childTwo)

    def replaceWithOffspring(self, parentOneID: int, parentTwoID: int, offspringOneChromosome: ndarray,
                             offspringTwoChromosome: ndarray) -> None:
        """Takes the offspring's alpha values and carries out the replacement strategy"""
        if self.replacementStrategy == "replaceParents":
            self.population[parentOneID].alphaValues = offspringOneChromosome
            self.population[parentOneID].resetOutputNetwork()
            self.population[parentTwoID].alphaValues = offspringTwoChromosome
            self.population[parentTwoID].resetOutputNetwork()
        elif self.replacementStrategy == "replaceWeakestTwo":
            weakestTwoIndividualIDs = self.getWeakestTwoIndividualIDs()
            self.population[weakestTwoIndividualIDs[0]].alphaValues = offspringOneChromosome
            self.population[weakestTwoIndividualIDs[0]].resetOutputNetwork()
            self.population[weakestTwoIndividualIDs[1]].alphaValues = offspringTwoChromosome
            self.population[weakestTwoIndividualIDs[1]].resetOutputNetwork()
        elif self.replacementStrategy == "replaceRandomTwo":
            randomTwoIndividualIDs = self.randomSelection()
            self.population[randomTwoIndividualIDs[0]].alphaValues = offspringOneChromosome
            self.population[randomTwoIndividualIDs[0]].resetOutputNetwork()
            self.population[randomTwoIndividualIDs[1]].alphaValues = offspringTwoChromosome
            self.population[randomTwoIndividualIDs[1]].resetOutputNetwork()
        else:
            print("ERROR - INVALID REPLACEMENT STRATEGY!!!")

    # ===================================================
    # ============== DAEMON UPDATE METHODS ==============
    # ===================================================
    def applyDaemonUpdate(self, individualID: int) -> None:
        """Applies the daemon update strategy to the individual"""
        # NOTE - All current daemon methods only work with the perEdge initialization strategy
        # TODO - Update to account for other initialization strategies
        print("Doing a daemon update to individual " + str(individualID) + "...")
        if self.daemonStrategy == "globalBinary":
            self.applyGlobalBinaryDaemon(individualID)
        elif self.daemonStrategy == "globalMean":
            self.applyGlobalMeanDaemon(individualID)
        elif self.daemonStrategy == "globalMedian":
            self.applyGlobalMedianDaemon(individualID)
        elif self.daemonStrategy == "personalMean":
            self.applyPersonalMeanDaemon(individualID)
        elif self.daemonStrategy == "personalMedian":
            self.applyPersonalMedianDaemon(individualID)
        else:
            print("ERROR - INVALID DAEMON STRATEGY!!!")

    def getAnnealedDaemonRate(self) -> float:
        """Returns a proportion (in [0, 1] if k=2) given the annealing schedule t = k*gen/(gen + maxGen)"""
        return self.annealingConstant * self.currentGeneration / (self.currentGeneration + self.numGenerations)

    def getDaemonUpdatedAlphaValue(self, currentAlpha: float, flowStatRatio=0.0) -> float:
        """Returns a new alpha value based on the current alpha, daemon strength, annealing schedule and flow-stat ratio"""
        # If the arc is greater than the statistic
        if flowStatRatio > 1.0:
            # Sample a Gaussian distribution
            nudgeVariance = currentAlpha * self.daemonStrength * flowStatRatio
            nudgedAlpha = random.gauss(currentAlpha, nudgeVariance)
            # Until the new value is strictly less (i.e. an increased discounting) than the old but not negative
            while nudgedAlpha > currentAlpha or nudgedAlpha < 0.0:
                nudgedAlpha = random.gauss(currentAlpha, nudgeVariance)
        # If the arc is less than the statistic
        elif flowStatRatio < 1.0:
            # Sample a Gaussian distribution
            nudgeVariance = currentAlpha * self.daemonStrength * (1 / flowStatRatio)
            nudgedAlpha = random.gauss(currentAlpha, nudgeVariance)
            # Until the new value is strictly greater (i.e. a decreased discounting) than the old
            while nudgedAlpha < currentAlpha:
                nudgedAlpha = random.gauss(currentAlpha, nudgeVariance)
        # Else block used on the global binary method (or if the flow stat ratio actually is zero)
        else:
            # Sample a Gaussian distribution
            nudgeVariance = currentAlpha * self.daemonStrength
            nudgedAlpha = random.gauss(currentAlpha, nudgeVariance)
            # Until the new value is strictly less (i.e. an increased discounting) than the old but not negative
            while nudgedAlpha > currentAlpha or nudgedAlpha < 0.0:
                nudgedAlpha = random.gauss(currentAlpha, nudgeVariance)
        return nudgedAlpha

    def applyGlobalBinaryDaemon(self, individualID: int) -> None:
        """Applies a daemon update that nudges alpha values for arcs opened in the global best solution"""
        individual = self.population[individualID]
        globalBestArcFlows = self.bestKnownSolution.arcFlows
        for edgeIndex in range(self.graph.numEdges):
            for arcIndex in range(self.graph.numArcsPerEdge):
                # If the global best has the arc unopened, do nothing
                if globalBestArcFlows[(edgeIndex, arcIndex)] == 0.0:
                    continue
                # Else get new alpha value based on annealed proportion
                else:
                    newAlpha = self.getDaemonUpdatedAlphaValue(individual.alphaValues[(edgeIndex, arcIndex)])
                    if self.initializationStrategy == "perEdge":
                        for cap in range(self.graph.numArcsPerEdge):
                            individual.alphaValues[(edgeIndex, cap)] = newAlpha
        individual.resetOutputNetwork()

    def applyGlobalMeanDaemon(self, individualID: int) -> None:
        """Applies a daemon update that nudges alpha values based on the mean assigned flow in the global best solution"""
        individual = self.population[individualID]
        globalBestArcFlows = self.bestKnownSolution.arcFlows
        # Find mean flow in the global best solution after removing all unopened arcs from the calculation
        arcFlowsArr = np.array(list(globalBestArcFlows.values()))
        arcFlowsArr[arcFlowsArr == 0.0] = np.nan
        globalMeanFlow = np.nanmean(arcFlowsArr)
        for edgeIndex in range(self.graph.numEdges):
            for arcIndex in range(self.graph.numArcsPerEdge):
                # If the global best has the arc unopened, do nothing
                if globalBestArcFlows[(edgeIndex, arcIndex)] == 0.0:
                    continue
                # Else get new alpha value based on annealed proportion
                else:
                    # Find the ratio of flow between this arc and the global best's mean arc flow
                    flowMeanRatio = globalBestArcFlows[(edgeIndex, arcIndex)] / globalMeanFlow
                    newAlpha = self.getDaemonUpdatedAlphaValue(individual.alphaValues[(edgeIndex, arcIndex)], flowStatRatio=flowMeanRatio)
                    if self.initializationStrategy == "perEdge":
                        for cap in range(self.graph.numArcsPerEdge):
                            individual.alphaValues[(edgeIndex, cap)] = newAlpha
        individual.resetOutputNetwork()

    def applyGlobalMedianDaemon(self, individualID: int) -> None:
        """Applies a daemon update that nudges alpha values based on the median assigned flow in the global best solution"""
        individual = self.population[individualID]
        globalBestArcFlows = self.bestKnownSolution.arcFlows
        # Find median flow in the global best solution after removing all unopened arcs from the calculation
        arcFlowsArr = np.array(list(globalBestArcFlows.values()))
        arcFlowsArr[arcFlowsArr == 0.0] = np.nan
        globalMedianFlow = np.nanmedian(arcFlowsArr)
        for edgeIndex in range(self.graph.numEdges):
            for arcIndex in range(self.graph.numArcsPerEdge):
                # If the global best has the arc unopened, do nothing
                if globalBestArcFlows[(edgeIndex, arcIndex)] == 0.0:
                    continue
                # Else get new alpha value based on annealed proportion
                else:
                    # Find the ratio of flow between this arc and the global best's median arc flow
                    flowMedianRatio = globalBestArcFlows[(edgeIndex, arcIndex)] / globalMedianFlow
                    newAlpha = self.getDaemonUpdatedAlphaValue(individual.alphaValues[(edgeIndex, arcIndex)], flowStatRatio=flowMedianRatio)
                    if self.initializationStrategy == "perEdge":
                        for cap in range(self.graph.numArcsPerEdge):
                            individual.alphaValues[(edgeIndex, cap)] = newAlpha
        individual.resetOutputNetwork()

    def applyPersonalMeanDaemon(self, individualID: int) -> None:
        """Applies a daemon update that nudges alpha values based on the mean assigned flow in the personal solution"""
        individual = self.population[individualID]
        # Find mean flow in the personal solution after removing all unopened arcs from the calculation
        arcFlowsArr = np.array(list(individual.arcFlows.values()))
        arcFlowsArr[arcFlowsArr == 0.0] = np.nan
        personalMeanFlow = np.nanmean(arcFlowsArr)
        for edgeIndex in range(self.graph.numEdges):
            for arcIndex in range(self.graph.numArcsPerEdge):
                # If the individual has the arc unopened, do nothing
                if individual.arcFlows[(edgeIndex, arcIndex)] == 0.0:
                    continue
                # Else get new alpha value based on annealed proportion
                else:
                    # Find the ratio of flow between this arc and the individual's mean arc flow
                    flowMeanRatio = individual.arcFlows[(edgeIndex, arcIndex)] / personalMeanFlow
                    newAlpha = self.getDaemonUpdatedAlphaValue(individual.alphaValues[(edgeIndex, arcIndex)],
                                                               flowStatRatio=flowMeanRatio)
                    if self.initializationStrategy == "perEdge":
                        for cap in range(self.graph.numArcsPerEdge):
                            individual.alphaValues[(edgeIndex, cap)] = newAlpha
        individual.resetOutputNetwork()

    def applyPersonalMedianDaemon(self, individualID: int) -> None:
        """Applies a daemon update that nudges alpha values based on the median assigned flow in the personal solution"""
        individual = self.population[individualID]
        # Find median flow in the personal solution after removing all unopened arcs from the calculation
        arcFlowsArr = np.array(list(individual.arcFlows.values()))
        arcFlowsArr[arcFlowsArr == 0.0] = np.nan
        personalMedianFlow = np.nanmedian(arcFlowsArr)
        for edgeIndex in range(self.graph.numEdges):
            for arcIndex in range(self.graph.numArcsPerEdge):
                # If the individual has the arc unopened, do nothing
                if individual.arcFlows[(edgeIndex, arcIndex)] == 0.0:
                    continue
                # Else get new alpha value based on annealed proportion
                else:
                    # Find the ratio of flow between this arc and the individual's median arc flow
                    flowMedianRatio = individual.arcFlows[(edgeIndex, arcIndex)] / personalMedianFlow
                    newAlpha = self.getDaemonUpdatedAlphaValue(individual.alphaValues[(edgeIndex, arcIndex)], flowStatRatio=flowMedianRatio)
                    if self.initializationStrategy == "perEdge":
                        for cap in range(self.graph.numArcsPerEdge):
                            individual.alphaValues[(edgeIndex, cap)] = newAlpha
        individual.resetOutputNetwork()

    # ==========================================================
    # ============== EVOLUTION STATISTICS METHODS ==============
    # ==========================================================
    def computeEvolutionStatistics(self) -> None:
        """Computes the population statistics for a given generation and logs them"""
        fitnessList = []
        for individual in self.population:
            fitnessList.append(individual.trueCost)
        fitnessArray = np.array(fitnessList, dtype='f')
        self.convergenceStats.append(np.min(fitnessArray))
        self.meanStats.append(np.mean(fitnessArray))
        self.medianStats.append(np.median(fitnessArray))
        self.stdDevStats.append(np.std(fitnessArray))

    def plotEvolutionStatistics(self, runID="") -> None:
        """Renders MatPlotLib graphs for each of the evolution statistics lists"""
        # Get generations and build figure/subplots
        generations = list(range(len(self.convergenceStats)))
        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Population Fitness Statistics over Generations")
        # Most Fit Individual Subplot
        axs[0, 0].plot(generations, self.convergenceStats, color="b")
        axs[0, 0].set_title("Most Fit")
        # Mean Fitness Subplot
        axs[0, 1].plot(generations, self.meanStats, color="r")
        axs[0, 1].set_title("Mean")
        # Std. Dev. Subplot
        axs[1, 0].plot(generations, self.stdDevStats, color="g")
        axs[1, 0].set_title("Std. Dev.")
        # Median Subplot
        axs[1, 1].plot(generations, self.medianStats, color="m")
        axs[1, 1].set_title("Median")
        # Add spacing
        plt.subplots_adjust(left=0.2,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.6,
                            hspace=0.4)
        # Save timestamped plot
        plt.savefig("GeneticEvoStats--" + runID + ".png")
        plt.close(fig)

    def logGenerationTimestamp(self, startTime: datetime) -> None:
        """Calculates the time, in seconds, since the evolution loop started and logs it"""
        currentTime = datetime.now()
        timeDiff = currentTime - startTime
        secondsElapsed = timeDiff.seconds + timeDiff.microseconds/1000000
        self.generationTimestamps.append(secondsElapsed)

    # ===============================================================
    # ============== HYPER-MUTATION/HILL CLIMB METHODS ==============
    # ===============================================================
    def solveWithNaiveHillClimb(self, printGenerations=True, drawing=True, drawLabels=False, isGraphing=True) -> FlowNetworkSolution:
        """Solves the population with a naive hill climb method"""
        # Initialize Population and Solve
        self.initializePopulation()
        self.solvePopulation()
        generation = 0
        if isGraphing is True:
            self.computeEvolutionStatistics()
        # Execute Hill Climb
        while self.isTerminated is not True:
            # Execute Hill Climb and Solve
            self.naiveHillClimb()
            self.solvePopulation()
            # Update Current Best Individual and Evaluate Termination
            bestIndividual = self.getMostFitIndividual()
            self.evaluateTermination(generation, bestIndividual.trueCost)
            # Compute Statistics
            if isGraphing is True:
                self.computeEvolutionStatistics()
            # Visualize & Print
            if printGenerations is True:
                print("Generation = " + str(generation) + "\tBest Individual = " + str(
                    bestIndividual.id) + "\tFitness = " + str(round(bestIndividual.trueCost, 2)))
            if drawing is True:
                self.visualizeBestIndividual(labels=drawLabels, leadingText="HC_Gen" + str(generation) + "_")
            generation += 1
        # Plot Statistics
        if isGraphing is True:
            self.plotEvolutionStatistics()
        # Return Best Solution Discovered
        return self.bestKnownSolution

    def naiveHillClimb(self) -> None:
        """Hypermutates all but the best individual"""
        sortedPopulation = self.rankPopulation()
        for i in range(1, self.populationSize):
            self.hypermutateIndividual(sortedPopulation[i].id)

    def hypermutateIndividual(self, individualNum: int) -> None:
        """Reinitializes the individual's entire alpha values (i.e. kills them off and spawns a new individual)"""
        individual = self.population[individualNum]
        newAlphas = self.getInitialAlphaValues()
        individual.setAlphaValues(newAlphas)
        individual.resetOutputNetwork()

    def hypermutatePopulation(self) -> None:
        """Reinitializes the entire population (i.e. an extinction event with a brand new population spawned)"""
        for individualNum in range(len(self.population)):
            self.hypermutateIndividual(individualNum)
