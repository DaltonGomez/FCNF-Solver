import copy
import random
import sys

import numpy as np
from numpy import ndarray

from src.AlphaGenetic.AlphaSolverPDLP import AlphaSolverPDLP
from src.AlphaGenetic.Individual import Individual
from src.Network.FlowNetwork import FlowNetwork
from src.Network.Solution import Solution
from src.Network.SolutionVisualizer import SolutionVisualizer


class Population:
    """Class that manages a population of alpha-relaxed individuals and handles their evolution with GA operators"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, network: FlowNetwork, minTargetFlow: float, populationSize=10, numGenerations=10):
        """Constructor of a Population instance"""
        # Input Attributes
        self.network = network
        self.minTargetFlow = minTargetFlow
        # Population & Solver Instance Objects
        self.population = []
        self.solver = AlphaSolverPDLP(self.network, self.minTargetFlow)  # Pre-builds variables/constraints on init
        self.isTerminated = False
        self.bestKnownSolution = None
        self.bestKnownCost = sys.maxsize

        # =======================
        # GA HYPERPARAMETERS
        # -----------------------
        # Globals & Initialization HPs
        self.populationSize = populationSize
        self.terminationMethod = "setGenerations"  # :param : "setGenerations", "stagnationPeriod"
        self.numGenerations = numGenerations
        self.stagnationPeriod = 5
        self.consecutiveStagnantGenerations = 0
        self.initializationDistribution = "uniform"  # :param : "uniform", "gaussian"
        self.initializationParams = [0.0, 1.0]  # :param: range if uniform distribution, mu and sigma if Gaussian
        # Individual Selection HPs
        self.selectionMethod = "tournament"  # :param : "tournament", "roulette", "random"
        self.tournamentSize = 3
        # Path Selection HPs
        self.pathSelectionMethod = "roulette"  # :param : "tournament", "roulette", "random", "top"
        self.pathRankingOrder = "most"  # :param : "most", "least"
        self.pathRankingMethod = "density"  # :param : "cost", "flow", "density", "length"
        self.pathSelectionSize = 2
        self.pathTournamentSize = 3
        # Crossover HPs
        self.crossoverMethod = "pathBased"  # :param : "onePoint", "twoPoint", "pathBased"
        self.crossoverRate = 1.0
        self.crossoverAttemptsPerGeneration = 1
        self.replacementStrategy = "replaceParents"  # : param : "replaceWeakestTwo", "replaceParents"
        # Mutation HPs
        self.mutationMethod = "pathBased"  # :param : "randomSingleArc", "randomSingleEdge", "randomTotal", "pathBased"
        self.mutationRate = 0.25
        self.nudgeParams = [0.0, 1.0]

    # ====================================================
    # ============== HYPERPARAMETER SETTERS ==============
    # ====================================================
    def setPopulationHyperparams(self, populationSize=10, terminationMethod="setGenerations", numGenerations=10,
                                 stagnationPeriod=5, initializationDistribution="uniform",
                                 initializationParams=(0.0, 1.0)) -> None:
        """Sets the GA class field that dictates the range when randomly initializing/updating alpha values \n
        :param int populationSize: Number of individuals in the GA population
        :param str terminationMethod: One of following: {"setGenerations", "stagnationPeriod"}
        :param int numGenerations: Number of iterations the population evolves for
        :param int stagnationPeriod: Number of stagnant consecutive generations needed for termination
        :param str initializationDistribution: One of following: {"uniform", "gaussian"}
        :param list initializationParams: Lower and upper bounds if uniform distribution; mu and sigma if Gaussian
        """
        self.populationSize = populationSize
        self.terminationMethod = terminationMethod
        self.numGenerations = numGenerations
        self.stagnationPeriod = stagnationPeriod
        self.initializationDistribution = initializationDistribution
        self.initializationParams = initializationParams

    def setIndividualSelectionHyperparams(self, selectionMethod="tournament", tournamentSize=3) -> None:
        """Sets the GA class fields that dictate how the selection of individuals is carried out \n
        :param str selectionMethod: One of following: {"tournament", "roulette", "random"}
        :param int tournamentSize: Size of tournament subset used if selectionMethod = "tournament"
        """
        self.selectionMethod = selectionMethod
        self.tournamentSize = tournamentSize

    def setPathSelectionHyperparams(self, pathSelectionMethod="roulette", pathRankingOrder="most",
                                    pathRankingMethod="density", pathSelectionSize=2, pathTournamentSize=3) -> None:
        """Sets the GA class fields that dictate how the selection of paths is carried out \n
        :param str pathSelectionMethod: One of following: {"tournament", "roulette", "random", "top"}
        :param str pathRankingOrder: One of following: {"most", "least"}
        :param str pathRankingMethod: One of following: {"cost", "flow", "density", "length"}
        :param int pathSelectionSize: Number of paths returned
        :param int pathTournamentSize: Size of tournament subset used if pathSelectionMethod = "tournament"
        """
        self.pathSelectionMethod = pathSelectionMethod
        self.pathRankingOrder = pathRankingOrder
        self.pathRankingMethod = pathRankingMethod
        self.pathSelectionSize = pathSelectionSize
        self.pathTournamentSize = pathTournamentSize

    def setCrossoverHyperparams(self, crossoverMethod="pathBased", replacementStrategy="replaceParents",
                                crossoverRate=1.0, crossoverAttemptsPerGeneration=1) -> None:
        """Sets the GA class fields that dictate how the crossover of individuals is carried out \n
        :param str crossoverMethod: One of following: {"onePoint", "twoPoint", "pathBased"}
        :param str replacementStrategy: One of following: {"replaceWeakestTwo", "replaceParents"}
        :param float crossoverRate: Probability in [0,1] that a crossover occurs
        :param int crossoverAttemptsPerGeneration: Number of attempted crossovers per generation
        """
        self.crossoverMethod = crossoverMethod
        self.crossoverRate = crossoverRate
        self.crossoverAttemptsPerGeneration = crossoverAttemptsPerGeneration
        self.replacementStrategy = replacementStrategy

    def setMutationHyperparams(self, mutationMethod="pathBased", mutationRate=0.25, nudgeParams=(0.0, 1.0)) -> None:
        """Sets the GA class fields that dictate how the mutation of individuals is carried out \n
        :param str mutationMethod: One of following: {"randomSingleArc", "randomSingleEdge", "randomTotal", "pathBasedRandom", "pathBasedNudge"}
        :param float mutationRate: Probability in [0,1] that a mutation occurs
        :param list nudgeParams: mu and sigma of the Gaussian distribution used for nudging
        """
        self.mutationMethod = mutationMethod
        self.mutationRate = mutationRate
        self.nudgeParams = nudgeParams

    # ============================================
    # ============== EVOLUTION LOOP ==============
    # ============================================
    def evolvePopulation(self, printGenerations=False, drawing=False, drawLabels=False) -> tuple:
        """Evolves the population for a specified number of generations"""
        # Initialize Population and Solve
        self.initializePopulation()
        self.solvePopulation()
        generation = 0
        # Evolve Population
        while self.isTerminated is not True:
            # Perform Operators and Solve
            self.selectAndCrossover()
            self.doMutations()
            self.solvePopulation()
            # Update Current Best Individual and Evaluate Termination
            bestIndividual = self.getMostFitIndividual()
            self.evaluateTermination(generation, bestIndividual.trueCost)
            # Visualize & Print
            if printGenerations is True:
                print("Generation = " + str(generation) + "\tBest Individual = " + str(
                    bestIndividual.id) + "\tFitness = " + str(round(bestIndividual.trueCost, 2)))
            if drawing is True:
                self.visualizeBestIndividual(labels=drawLabels, leadingText="Gen" + str(generation) + "_")
            generation += 1
        # Return Best Solution Discovered
        return self.bestKnownCost, self.bestKnownSolution

    def selectAndCrossover(self) -> None:
        """Performs the selection and crossover at each generation"""
        random.seed()
        for n in range(self.crossoverAttemptsPerGeneration):
            if random.random() < self.crossoverRate:
                parents = self.selectIndividuals()
                if self.crossoverMethod == "pathBased":
                    parentOnePaths = self.selectPaths(parents[0])
                    parentTwoPaths = self.selectPaths(parents[1])
                    self.crossover(parents[0], parents[1], parentOnePaths=parentOnePaths, parentTwoPaths=parentTwoPaths)
                else:
                    self.crossover(parents[0], parents[1])

    def doMutations(self) -> None:
        """Performs mutations on individuals given a mutation probability"""
        random.seed()
        for individualID in range(self.populationSize):
            individual = self.population[individualID]
            if individual.isSolved is False:
                continue
            if random.random() < self.mutationRate:
                if self.mutationMethod == "pathBased":
                    individualPaths = self.selectPaths(individualID)
                    self.mutate(individualID, selectedPaths=individualPaths)
                else:
                    self.mutate(individualID)

    def evaluateTermination(self, generation: int, newBestCost: float) -> None:
        """Checks for termination using the given method and updates the best known solution"""
        if self.terminationMethod == "setGenerations":
            if newBestCost < self.bestKnownCost:
                self.bestKnownCost = newBestCost
                self.bestKnownSolution = self.writeIndividualsSolution(self.getMostFitIndividual())
            if generation >= self.numGenerations:
                self.isTerminated = True
        elif self.terminationMethod == "stagnationPeriod":
            if newBestCost < self.bestKnownCost:
                self.bestKnownCost = newBestCost
                self.bestKnownSolution = self.writeIndividualsSolution(self.getMostFitIndividual())
                self.consecutiveStagnantGenerations = 0
            elif newBestCost >= self.bestKnownCost:
                self.consecutiveStagnantGenerations += 1
                if self.consecutiveStagnantGenerations >= self.stagnationPeriod:
                    self.isTerminated = True

    def solveWithNaiveHillClimb(self, printGenerations=False, drawing=False, drawLabels=False) -> tuple:
        """Solves the population with a naive hill climb method"""
        # Initialize Population and Solve
        self.initializePopulation()
        self.solvePopulation()
        generation = 0
        # Execute Hill Climb
        while self.isTerminated is not True:
            # Execute Hill Climb and Solve
            self.naiveHillClimb()
            self.solvePopulation()
            # Update Current Best Individual and Evaluate Termination
            bestIndividual = self.getMostFitIndividual()
            self.evaluateTermination(generation, bestIndividual.trueCost)
            # Visualize & Print
            if printGenerations is True:
                print("Generation = " + str(generation) + "\tBest Individual = " + str(
                    bestIndividual.id) + "\tFitness = " + str(round(bestIndividual.trueCost, 2)))
            if drawing is True:
                self.visualizeBestIndividual(labels=drawLabels, leadingText="Gen" + str(generation) + "_")
            generation += 1
        # Return Best Solution Discovered
        return self.bestKnownCost, self.bestKnownSolution

    # ====================================================
    # ============== INITIALIZATION METHODS ==============
    # ====================================================
    def initializePopulation(self) -> None:
        """Initializes the GA population with random alpha values"""
        for individual in range(self.populationSize):
            thisGenotype = self.getInitialAlphaValues()
            thisIndividual = Individual(individual, self.network, thisGenotype)
            self.population.append(thisIndividual)

    def getInitialAlphaValues(self) -> ndarray:
        """Returns a randomly initialized array of alpha values (i.e. the genotype)"""
        tempAlphaValues = []
        for edge in range(self.network.numEdges):
            tempEdge = []
            for cap in range(self.network.numArcCaps):
                thisAlphaValue = self.getAlphaValue()
                tempEdge.append(thisAlphaValue)
            tempAlphaValues.append(tempEdge)
        initialGenotype = np.array(tempAlphaValues)
        return initialGenotype

    def getAlphaValue(self) -> float:
        """Returns a single alpha value for population initialization"""
        random.seed()
        randomGene = 1.0
        if self.initializationDistribution == "uniform":
            randomGene = random.uniform(self.initializationParams[0], self.initializationParams[1])
        elif self.initializationDistribution == "gaussian":
            randomGene = random.gauss(self.initializationParams[0], self.initializationParams[1])
        return randomGene

    # ===============================================================
    # ============== HYPER-MUTATION/HILL CLIMB METHODS ==============
    # ===============================================================
    def hypermutateIndividual(self, individualNum: int) -> None:
        """Reinitializes the individual's entire alpha values (i.e. kills them off and spawns a new individual)"""
        individual = self.population[individualNum]
        newAlphas = self.getInitialAlphaValues()
        individual.setAlphaValues(newAlphas)
        individual.resetOutputNetwork()

    def hypermutatePopulation(self) -> None:
        """Reinitializes the entire population (i.e. an extinction event with a brand new population spawned)"""
        for individual in self.population:
            self.hypermutateIndividual(individual)

    def naiveHillClimb(self) -> None:
        """Hypermutates all but the best individual"""
        sortedPopulation = self.rankPopulation()
        for i in range(1, self.populationSize):
            self.hypermutateIndividual(sortedPopulation[i].id)

    # =============================================
    # ============== RANKING METHODS ==============
    # =============================================
    def rankPopulation(self) -> list:
        """Ranks the population not in place in ascending order of cost (i.e. Lower cost -> More fit) and returns"""
        # NOTE: The population should never be sorted in-place as this will cause bugs in the execution of operators
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
        for individual in self.population:
            if individual.isSolved is False:
                self.solveIndividual(individual)

    def solveIndividual(self, individual: Individual) -> None:
        """Solves a single individual and writes the expressed network to the individual"""
        # Overwrite new objective function with new alpha values and solve
        self.solver.updateObjectiveFunction(individual.alphaValues)
        self.solver.solveModel()
        # Write expressed network output data to individual
        individual.isSolved = True
        individual.arcFlows = self.solver.getArcFlowsDict()
        individual.arcsOpened = self.solver.getArcsOpenDict()
        individual.srcFlows = self.solver.getSrcFlowsList()
        individual.sinkFlows = self.solver.getSinkFlowsList()
        individual.trueCost = self.solver.calculateTrueCost()
        individual.fakeCost = self.solver.getObjectiveValue()
        # Reset solver
        self.solver.resetSolver()

    def writeIndividualsSolution(self, individual: Individual) -> Solution:
        """Writes the individual's output to as solution object"""
        solution = None
        if individual.isSolved is True:
            solution = Solution(self.network, self.minTargetFlow, individual.fakeCost, individual.trueCost,
                                individual.srcFlows, individual.sinkFlows, individual.arcFlows, individual.arcsOpened,
                                "alphaGA", False, self.network.isSourceSinkCapacitated,
                                self.network.isSourceSinkCharged)
        else:
            print("An unsolved individual cannot write a solution!")
        return solution

    def resetOutputFields(self) -> None:
        """Resets the output fields stored in the population"""
        self.isTerminated = False
        self.population = []
        self.solver = AlphaSolverPDLP(self.network, self.minTargetFlow)  # Pre-builds variables/constraints on init
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
            visualizer.drawGraphWithLabels(leadingText=leadingText)
        else:
            visualizer.drawUnlabeledGraph(leadingText=leadingText)

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

    def printBestIndividualsPaths(self) -> None:
        """Prints the path data of the best individual"""
        bestIndividual = self.getMostFitIndividual()
        bestIndividual.computeAllUsedPaths()
        bestIndividual.printAllPaths()

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
        return selectedIndividualIDs

    def randomSelection(self) -> list:
        """Returns a random subset of individuals in the population (w/o replacement)"""
        random.seed()
        randomIndividualsIDs = random.sample(range(len(self.population)), 2)
        return randomIndividualsIDs

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
        selection = []
        for i in range(2):
            topPick = tournament.pop(0)
            selection.append(topPick[0])
        return selection

    # ============================================================
    # ============== PATH SELECTION OPERATORS ====================
    # ============================================================
    def selectPaths(self, individualID: int) -> list:
        """Hyper-selection operator that calls specific path selection method based on hyperparameters \n"""
        selectedPaths = []
        # If paths are selected randomly, skip the ranking process
        if self.pathSelectionMethod == "random":
            selectedPaths = self.randomPathSelection(individualID)
        # Else rank paths as a set of tuples based on self.pathRankingMethod and self.pathRankingOrder
        else:
            individual = self.population[individualID]
            rankedPaths = []
            # Compute paths and resize return selection length if necessary
            if len(individual.paths) == 0:
                individual.computeAllUsedPaths()
                # If the number of paths is still zero, return an empty list
                if len(individual.paths) == 0:
                    return []
            # Adjust the number of returned paths to either the hyperparameter or the total number of paths (if smaller)
            thisSelectionSize = self.pathSelectionSize
            if self.pathSelectionSize > len(individual.paths):
                thisSelectionSize = len(individual.paths)
            # Rank paths based on self.pathRankingMethod and self.pathRankingOrder hyperparameters
            if self.pathRankingMethod == "cost":
                if self.pathRankingOrder == "most":
                    for path in individual.paths:
                        rankedPaths.append((path, path.pathRoutingCost))
                    rankedPaths.sort(key=lambda p: p[1], reverse=True)
                elif self.pathRankingOrder == "least":
                    for path in individual.paths:
                        rankedPaths.append((path, path.pathRoutingCost))
                    rankedPaths.sort(key=lambda p: p[1], reverse=False)
            elif self.pathRankingMethod == "flow":
                if self.pathRankingOrder == "most":
                    for path in individual.paths:
                        rankedPaths.append((path, path.pathFlow))
                    rankedPaths.sort(key=lambda p: p[1], reverse=True)
                elif self.pathRankingOrder == "least":
                    for path in individual.paths:
                        rankedPaths.append((path, path.pathFlow))
                    rankedPaths.sort(key=lambda p: p[1], reverse=False)
            elif self.pathRankingMethod == "density":
                if self.pathRankingOrder == "most":
                    for path in individual.paths:
                        rankedPaths.append((path, path.flowPerRoutingCost))
                    rankedPaths.sort(key=lambda p: p[1], reverse=True)
                elif self.pathRankingOrder == "least":
                    for path in individual.paths:
                        rankedPaths.append((path, path.flowPerRoutingCost))
                    rankedPaths.sort(key=lambda p: p[1], reverse=False)
            elif self.pathRankingMethod == "length":
                if self.pathRankingOrder == "most":
                    for path in individual.paths:
                        rankedPaths.append((path, path.length))
                    rankedPaths.sort(key=lambda p: p[1], reverse=True)
                elif self.pathRankingOrder == "least":
                    for path in individual.paths:
                        rankedPaths.append((path, path.length))
                    rankedPaths.sort(key=lambda p: p[1], reverse=False)
            # Generate selected paths with method call based on pathSelectionMethod
            if self.pathSelectionMethod == "top":
                selectedPaths = self.topPathSelection(rankedPaths, thisSelectionSize)
            elif self.pathSelectionMethod == "roulette":
                selectedPaths = self.rouletteWheelPathSelection(rankedPaths, thisSelectionSize)
            elif self.pathSelectionMethod == "tournament":
                selectedPaths = self.tournamentPathSelection(rankedPaths, thisSelectionSize)
        return selectedPaths

    def randomPathSelection(self, individualID: int) -> list:
        """Returns a random subset of paths in an individual (w/o replacement)"""
        random.seed()
        individual = self.population[individualID]
        # Compute paths and resize return selection length if necessary
        if len(individual.paths) == 0:
            individual.computeAllUsedPaths()
            # If the number of paths is still zero, return an empty list
            if len(individual.paths) == 0:
                return []
        thisSelectionSize = self.pathSelectionSize
        if self.pathSelectionSize > len(individual.paths):
            thisSelectionSize = len(individual.paths)
        # Randomly sample the paths list
        selectedPaths = random.sample(individual.paths, thisSelectionSize)
        return selectedPaths

    @staticmethod
    def topPathSelection(rankedPaths: list, thisSelectionSize: int) -> list:
        """Returns the n top paths for the individual based on the path ranking method"""
        # Select paths subset from beginning of full path list
        selectedPaths = []
        for path in range(thisSelectionSize):
            selectedPaths.append(rankedPaths[path][0])
        return selectedPaths

    @staticmethod
    def rouletteWheelPathSelection(rankedPaths: list, thisSelectionSize: int) -> list:
        """Selects paths from an individual probabilistically by their normalized fitness (based on path ranking method)"""
        random.seed()
        # Build cumulative probability function
        cumulativeFitness = 0
        for p in range(len(rankedPaths)):
            cumulativeFitness += rankedPaths[p][1]
        cumulativeProbabilities = [rankedPaths[0][1] / cumulativeFitness]
        for i in range(1, len(rankedPaths)):
            cumulativeProbabilities.append(
                (rankedPaths[i][1] / cumulativeFitness) + cumulativeProbabilities[i - 1])
        # Build selected paths set
        selectedPaths = []
        duplicateCheckingSet = set()
        while len(selectedPaths) < thisSelectionSize:
            rng = random.random()
            for p in range(len(rankedPaths)):
                if rng < cumulativeProbabilities[p]:
                    # Utilize a tuple of the nodes as a hashable id to prevent duplication
                    hashableID = tuple(rankedPaths[p][0].nodes)
                    if hashableID not in duplicateCheckingSet:
                        selectedPaths.append(rankedPaths[p][0])
                        duplicateCheckingSet.add(hashableID)
                        break
        return selectedPaths

    def tournamentPathSelection(self, rankedPaths: list, thisSelectionSize: int) -> list:
        """Selects the best k paths out of a randomly chosen subset of size n"""
        random.seed()
        # Select random subset of paths
        thisTournamentSize = self.pathTournamentSize
        if self.pathTournamentSize > len(rankedPaths):
            thisTournamentSize = len(rankedPaths)
        tournament = random.sample(rankedPaths, thisTournamentSize)
        # Sort the tournament based on most or least direction
        if self.pathRankingOrder[0] == "m":
            tournament.sort(key=lambda p: p[1], reverse=True)
        elif self.pathRankingOrder[0] == "l":
            rankedPaths.sort(key=lambda p: p[1], reverse=False)
        # Select top and return
        selectedPaths = []
        if len(tournament) < thisSelectionSize:
            thisSelectionSize = len(tournament)
        for i in range(thisSelectionSize):
            pathTuple = tournament.pop(0)
            selectedPaths.append(pathTuple[0])
        return selectedPaths

    # ==============================================
    # ============== MUTATION METHODS ==============
    # ==============================================
    def mutate(self, individualID: int, selectedPaths=None) -> None:
        """Hyper-selection operator that calls specific mutation method based on hyperparameters"""
        if selectedPaths is None:
            selectedPaths = []
        if self.mutationMethod == "randomTotal":
            self.hypermutateIndividual(individualID)
        elif self.mutationMethod == "randomSingleArc":
            self.randomSingleArcMutation(individualID)
        elif self.mutationMethod == "randomSingleEdge":
            self.randomSingleEdgeMutation(individualID)
        elif self.mutationMethod == "pathBasedRandom":
            self.selectedPathsRandomMutation(individualID, selectedPaths)
        elif self.mutationMethod == "pathBasedNudge":
            self.selectedPathsNudgeMutation(individualID, selectedPaths)

    def randomSingleArcMutation(self, individualNum: int) -> None:
        """Mutates an individual at only one random arc in the chromosome"""
        random.seed()
        mutatedEdge = random.randint(0, self.network.numEdges - 1)
        mutatedCap = random.randint(0, self.network.numArcCaps - 1)
        individual = self.population[individualNum]
        individual.alphaValues[mutatedEdge][mutatedCap] = self.getAlphaValue()
        individual.resetOutputNetwork()

    def randomSingleEdgeMutation(self, individualNum: int) -> None:
        """Mutates an individual at all arcs in a random edge in the chromosome"""
        random.seed()
        mutatedEdge = random.randint(0, self.network.numEdges - 1)
        individual = self.population[individualNum]
        for arcIndex in range(self.network.numArcCaps):
            individual.alphaValues[mutatedEdge][arcIndex] = self.getAlphaValue()
        individual.resetOutputNetwork()

    def selectedPathsRandomMutation(self, individualNum: int, selectedPaths: list) -> None:
        """Randomly mutates all arcs for each edge in the selected paths of an individual"""
        individual = self.population[individualNum]
        for path in selectedPaths:
            for edge in path.edges:
                edgeIndex = self.network.edgesDict[edge]
                for arcIndex in range(self.network.numArcCaps):
                    individual.alphaValues[edgeIndex][arcIndex] = self.getAlphaValue()
        individual.resetOutputNetwork()

    def selectedPathsNudgeMutation(self, individualNum: int, selectedPaths: list) -> None:
        """Nudges all arcs for each edge in the selected paths of an individual"""
        random.seed()
        nudgeMagnitude = random.gauss(self.nudgeParams[0], self.nudgeParams[1])
        individual = self.population[individualNum]
        for path in selectedPaths:
            for edge in path.edges:
                edgeIndex = self.network.edgesDict[edge]
                for arcIndex in range(self.network.numArcCaps):
                    individual.alphaValues[edgeIndex][arcIndex] = individual.alphaValues[edgeIndex][
                                                                      arcIndex] + nudgeMagnitude
                    if individual.alphaValues[edgeIndex][arcIndex] < 0.0:
                        individual.alphaValues[edgeIndex][arcIndex] = 0.0
        individual.resetOutputNetwork()

    # =================================================
    # ============== CROSSOVER OPERATORS ==============
    # =================================================
    def crossover(self, parentOneID: int, parentTwoID: int, parentOnePaths=None, parentTwoPaths=None) -> None:
        """Hyper-selection operator that calls specific crossover method based on hyperparameters"""
        if parentTwoPaths is None:
            parentTwoPaths = []
        if parentOnePaths is None:
            parentOnePaths = []
        if self.selectionMethod == "onePoint":
            self.randomOnePointCrossover(parentOneID, parentTwoID)
        elif self.selectionMethod == "twoPoint":
            self.randomTwoPointCrossover(parentOneID, parentTwoID)
        elif self.selectionMethod == "pathBased":
            self.pathBasedCrossover(parentOneID, parentTwoID, parentOnePaths, parentTwoPaths)

    def randomOnePointCrossover(self, parentOneID: int, parentTwoID: int) -> None:
        """Crossover of 2 chromosomes at a single random point\n
        :param int parentOneID: Index of first parent in population
        :param int parentTwoID: Index of second parent in population
        """
        random.seed()
        # Generate crossover point
        crossoverPoint = random.randint(1, self.network.numEdges - 2)
        parentOneChromosome = self.population[parentOneID].alphaValues
        parentTwoChromosome = self.population[parentTwoID].alphaValues
        # Create new offspring chromosomes
        offspringOneChromosome = np.zeros((self.network.numEdges, self.network.numArcCaps))
        offspringTwoChromosome = np.zeros((self.network.numEdges, self.network.numArcCaps))
        # Up to crossover point
        for edge in range(crossoverPoint):
            for cap in range(self.network.numArcCaps):
                offspringOneChromosome[edge][cap] = parentOneChromosome[edge][cap]
                offspringTwoChromosome[edge][cap] = parentTwoChromosome[edge][cap]
        # After crossover point
        for edge in range(crossoverPoint, self.network.numEdges):
            for cap in range(self.network.numArcCaps):
                offspringOneChromosome[edge][cap] = parentTwoChromosome[edge][cap]
                offspringTwoChromosome[edge][cap] = parentOneChromosome[edge][cap]
        # Do replacement with offspring
        self.replaceWithOffspring(parentOneID, parentTwoID, offspringOneChromosome, offspringTwoChromosome)

    def randomTwoPointCrossover(self, parentOneID: int, parentTwoID: int) -> None:
        """Crossover of 2 chromosomes at a two random points\n
        :param int parentOneID: Index of first parent in population
        :param int parentTwoID: Index of second parent in population
        """
        random.seed()
        # Generate crossover points
        crossoverPointOne = random.randint(0, self.network.numEdges - 3)
        crossoverPointTwo = random.randint(crossoverPointOne, self.network.numEdges - 1)
        parentOneChromosome = self.population[parentOneID].alphaValues
        parentTwoChromosome = self.population[parentTwoID].alphaValues
        # Create new offspring chromosomes
        offspringOneChromosome = np.zeros((self.network.numEdges, self.network.numArcCaps))
        offspringTwoChromosome = np.zeros((self.network.numEdges, self.network.numArcCaps))
        # Up to first point
        for edge in range(crossoverPointOne):
            for cap in range(self.network.numArcCaps):
                offspringOneChromosome[edge][cap] = parentOneChromosome[edge][cap]
                offspringTwoChromosome[edge][cap] = parentTwoChromosome[edge][cap]
        # From point one to point two
        for edge in range(crossoverPointOne, crossoverPointTwo):
            for cap in range(self.network.numArcCaps):
                offspringOneChromosome[edge][cap] = parentTwoChromosome[edge][cap]
                offspringTwoChromosome[edge][cap] = parentOneChromosome[edge][cap]
        # From point two to the end
        for edge in range(crossoverPointTwo, self.network.numEdges):
            for cap in range(self.network.numArcCaps):
                offspringOneChromosome[edge][cap] = parentOneChromosome[edge][cap]
                offspringTwoChromosome[edge][cap] = parentTwoChromosome[edge][cap]
        # Do replacement with offspring
        self.replaceWithOffspring(parentOneID, parentTwoID, offspringOneChromosome, offspringTwoChromosome)

    def pathBasedCrossover(self, parentOneID: int, parentTwoID: int, parentOnePaths: list,
                           parentTwoPaths: list) -> None:
        """Crossover based on the flow per cost density of paths of the parents\n
        :param int parentOneID: Index of first parent in population
        :param int parentTwoID: Index of second parent in population
        :param list parentOnePaths: List of paths to be crossed-over from parent 1 to parent 2 (Can be any length)
        :param list parentTwoPaths: List of paths to be crossed-over from parent 2 to parent 1 (Can be any length)
        """
        parentOneChromosome = self.population[parentOneID].alphaValues
        parentTwoChromosome = self.population[parentTwoID].alphaValues
        offspringOneChromosome = copy.deepcopy(parentOneChromosome)
        offspringTwoChromosome = copy.deepcopy(parentTwoChromosome)
        # Crossover values in Parent 1's Path
        for path in parentOnePaths:
            for edge in path.edges:
                edgeIndex = self.network.edgesDict[edge]
                for cap in range(self.network.numArcCaps):
                    offspringOneChromosome[edgeIndex][cap] = parentTwoChromosome[edgeIndex][cap]
                    offspringTwoChromosome[edgeIndex][cap] = parentOneChromosome[edgeIndex][cap]
        # Crossover values in Parent 2's Path
        for path in parentTwoPaths:
            for edge in path.edges:
                edgeIndex = self.network.edgesDict[edge]
                for cap in range(self.network.numArcCaps):
                    offspringOneChromosome[edgeIndex][cap] = parentTwoChromosome[edgeIndex][cap]
                    offspringTwoChromosome[edgeIndex][cap] = parentOneChromosome[edgeIndex][cap]
        # Do replacement with offspring
        self.replaceWithOffspring(parentOneID, parentTwoID, offspringOneChromosome, offspringTwoChromosome)

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
