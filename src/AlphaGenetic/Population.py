import copy
import random

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

        # =======================
        # GA HYPERPARAMETERS
        # -----------------------
        # Globals & Initialization HPs
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.initializationDistribution = "uniform"  # :param : "uniform", "gaussian"
        self.initializationParams = [0.0,
                                     1.0]  # :param: Lower and upper bounds if uniform distribution, or mu and sigma if Gaussian
        # Individual Selection HPs (For Crossover Exclusively)
        self.selectionMethod = "tournament"  # :param : "tournament", "roulette", "top", "random"
        self.tournamentSize = 2
        # Path Selection HPs
        self.pathSelectionMethod = "roulette"  # :param : "tournament", "roulette", "top", "random"
        self.pathRankingMethod = "leastDense"  # :param : "mostDense", "leastDense", "mostFlow", "leastFlow", "mostCost", "leastCost", "mostEdges", "leastEdges"
        self.pathSelectionSize = 2
        self.pathTournamentSize = 2
        # Crossover HPs
        self.crossoverMethod = "pathBased"  # :param : "onePoint", "twoPoint", "pathBased"
        self.crossoverRate = 0.90
        self.crossoverAttemptsPerGeneration = 1
        self.replacementStrategy = "replaceWeakestTwo"  # : param : "replaceWeakestTwo", "replaceParents"
        # Mutation HPs
        self.mutationMethod = "pathBased"  # :param : "randomSingle", "randomTotal", "pathBased"
        self.mutationRate = 0.50

        # Initialize Population
        self.initializePopulation()

    # ==================================================================
    # ============== HYPERPARAMETER SETTERS & INITIALIZER ==============
    # ==================================================================
    def setGlobalHyperparams(self, populationSize: int, numGenerations: int, initializationDistribution: str,
                             initializationParams: list) -> None:
        """Sets the GA class field that dictates the range when randomly initializing/updating alpha values \n
        :param int populationSize: Number of individuals in the GA population
        :param int numGenerations: Number of iterations the population evolves for
        :param str initializationDistribution: One of following: {"uniform", "gaussian"}
        :param list initializationParams: Lower and upper bounds if uniform distribution, or mu and sigma if Gaussian
        """
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.initializationDistribution = initializationDistribution
        self.initializationParams = initializationParams

    def setIndividualSelectionHyperparams(self, selectionMethod: str, tournamentSize: int) -> None:
        """Sets the GA class fields that dictate how the selection of individuals is carried out \n
        :param str selectionMethod: One of following: {"tournament", "roulette", "top", "random"}
        :param int tournamentSize: Size of tournament subset selected if selectionMethod = "tournament"
        """
        self.selectionMethod = selectionMethod
        self.tournamentSize = tournamentSize

    def setPathSelectionHyperparams(self, selectionMethod: str, pathRankingMethod: str, selectionSize: int,
                                    tournamentSize: int) -> None:
        """Sets the GA class fields that dictate how the selection of paths is carried out \n
        :param str selectionMethod: One of following: {"tournament", "roulette", "top", "random"}
        :param str pathRankingMethod: One of following: {"mostDense", "leastDense", "mostFlow", "leastFlow", "mostCost", "leastCost", "mostEdges", "leastEdges"}
        :param int selectionSize: Number of individuals returned
        :param int tournamentSize: Size of tournament subset selected if selectionMethod = "tournament"
        """
        self.pathSelectionMethod = selectionMethod
        self.pathRankingMethod = pathRankingMethod
        self.pathSelectionSize = selectionSize
        self.pathTournamentSize = tournamentSize

    def setCrossoverHyperparams(self, crossoverMethod: str, replacementStrategy: str, crossoverRate: float,
                                crossoverAttemptsPerGeneration: int) -> None:
        """Sets the GA class fields that dictate how the crossover of individuals is carried out \n
        :param str crossoverMethod: One of following: {"onePoint", "twoPoint", "pathBased"}
        :param int replacementStrategy: One of following: {"replaceWeakestTwo", "replaceParents"}
        :param str crossoverRate: Probability in [0,1] that a crossover occurs
        :param int crossoverAttemptsPerGeneration: Number of attempted crossovers per generation
        """
        self.crossoverMethod = crossoverMethod
        self.crossoverRate = crossoverRate
        self.crossoverAttemptsPerGeneration = crossoverAttemptsPerGeneration
        self.replacementStrategy = replacementStrategy

    def setMutationHyperparams(self, mutationMethod: str, mutationRate: float) -> None:
        """Sets the GA class fields that dictate how the mutation of individuals is carried out \n
        :param str mutationMethod: One of following: {"randomSingle", "randomTotal", "pathBased"}
        :param str mutationRate: Probability in [0,1] that a mutation occurs
        """
        self.mutationMethod = mutationMethod  # :param : "randomSingle", "randomTotal", "pathBased"
        self.mutationRate = mutationRate

    # ============================================
    # ============== EVOLUTION LOOP ==============
    # ============================================
    def evolvePopulation(self, drawing=False, drawLabels=False) -> None:
        """Evolves the population for a specified number of generations"""
        for generation in range(self.numGenerations):
            # TODO - IMPLEMENT SELECTION & CROSSOVER
            # TODO - IMPLEMENT SELECTION & MUTATION
            self.solvePopulation()
            if drawing is True:
                self.visualizeBestIndividual(labels=drawLabels, leadingText="Gen" + str(generation) + "_")
            print("Generation = " + str(generation) + "\tBest Individual = " + str(self.population[0].trueCost))

    def solveWithNaiveHillClimb(self, drawing=False, drawLabels=False) -> None:
        """Evolves the population for a specified number of generations"""
        for generation in range(self.numGenerations):
            # Solve and visualize
            self.naiveHillClimb()
            self.solvePopulation()
            if drawing is True:
                self.visualizeBestIndividual(labels=drawLabels, leadingText="Gen" + str(generation) + "_")
            self.printBestIndividualsPaths()
            print("Generation = " + str(generation) + "\tBest Individual = " + str(self.population[0].trueCost))

    # ====================================================
    # ============== INITIALIZATION METHODS ==============
    # ====================================================
    def initializePopulation(self) -> None:
        """Initializes the GA population with random alpha values"""
        for individual in range(self.populationSize):
            thisGenotype = self.getInitialAlphaValues()
            thisIndividual = Individual(self.network, thisGenotype)
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

    # =========================================================
    # ============== MUTATION/HILL CLIMB METHODS ==============
    # =========================================================
    def randomSingleMutation(self, individualNum: int) -> None:
        """Mutates an individual at only one random gene in the chromosome"""
        random.seed()
        mutatedEdge = random.randint(0, self.network.numEdges - 1)
        mutatedCap = random.randint(0, self.network.numArcCaps - 1)
        individual = self.population[individualNum]
        individual.alphaValues[mutatedEdge][mutatedCap] = self.getAlphaValue()
        individual.resetOutputNetwork()

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
        """Sorts the population by rank and hypermutates the non-best individuals only at each generation"""
        self.population = self.rankPopulation()
        for i in range(1, self.populationSize):
            self.hypermutateIndividual(i)

    # ============================================
    # ============== HELPER METHODS ==============
    # ============================================
    def rankPopulation(self) -> list:
        """Ranks the population in ascending order of true cost (i.e. Lower cost -> More fit) and returns"""
        sortedPopulation = sorted(self.population, key=lambda x: x.trueCost)
        return sortedPopulation

    def getMostFitIndividual(self) -> Individual:
        """Returns the most fit individual in the population"""
        sortedPop = self.rankPopulation()
        return sortedPop[0]

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

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
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

    def visualizeIndividual(self, individual: Individual, labels=False, leadingText="") -> None:
        """Renders the visualization for a specified individual"""
        solution = Solution(self.network, self.minTargetFlow, individual.fakeCost, individual.trueCost,
                            individual.srcFlows, individual.sinkFlows, individual.arcFlows, individual.arcsOpened,
                            "alphaGA", False, self.network.isSourceSinkCapacitated, self.network.isSourceSinkCharged)
        visualizer = SolutionVisualizer(solution)
        if labels is True:
            visualizer.drawGraphWithLabels(leadingText=leadingText)
        else:
            visualizer.drawUnlabeledGraph(leadingText=leadingText)

    def printBestIndividualsPaths(self) -> None:
        """Prints the path data of the best individual"""
        bestIndividual = self.getMostFitIndividual()
        bestIndividual.computeAllUsedPaths()
        bestIndividual.printAllPaths()

    # TODO - !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO - REVISE HERE DOWN (ALL OLD METHODS)
    # TODO - !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ============================================================
    # ============== INDIVIDUAL SELECTION OPERATORS ==============
    # ============================================================
    def selectIndividuals(self) -> list:
        """Hyper-selection operator that calls specific selection method based on hyperparameters"""
        selectedIndividualIDs = []
        if self.selectionMethod == "tournament":
            selectedIndividualIDs = self.tournamentSelection(self.tournamentSize)
        elif self.selectionMethod == "roulette":
            selectedIndividualIDs = self.rouletteWheelSelection()
        elif self.selectionMethod == "top":
            selectedIndividualIDs = self.topSelection()
        elif self.selectionMethod == "random":
            selectedIndividualIDs = self.randomSelection()
        return selectedIndividualIDs

    def randomSelection(self) -> list:
        """Returns a random subset of individuals in the population (w/o replacement)"""
        random.seed()
        populationIDs = []
        for i in range(len(self.population)):
            populationIDs.append(i)
        randomIndividualsIDs = random.sample(populationIDs, 2)
        return randomIndividualsIDs

    def topSelection(self) -> list:
        """Returns the top n individuals in the population"""
        self.rankPopulation()
        topIndividualIDs = []
        for i in range(2):
            topIndividualIDs.append(i)
        return topIndividualIDs

    def rouletteWheelSelection(self) -> list:
        """Selects individuals probabilistically by their normalized fitness"""
        random.seed()
        self.rankPopulation()
        fitnessFromCost = []
        for individual in range(len(self.population)):
            fitnessFromCost.append(1 / self.population[individual].trueCost)
        cumulativeFitness = 0
        for individual in range(len(self.population)):
            cumulativeFitness += fitnessFromCost[individual]
        cumulativeProbabilities = [fitnessFromCost[0] / cumulativeFitness]
        for i in range(1, len(self.population)):
            cumulativeProbabilities.append(
                (fitnessFromCost[i] / cumulativeFitness) + cumulativeProbabilities[i - 1])
        selectionSet = set()
        while len(selectionSet) < 2:
            rng = random.random()
            for individual in range(len(self.population)):
                if rng < cumulativeProbabilities[individual]:
                    selectionSet.add(individual)
                    break
        return list(selectionSet)

    def tournamentSelection(self, tournamentSize: int) -> list:
        """Selects the best k individuals out of a randomly chosen subset of size n"""
        random.seed()
        # Select subset of population
        populationIDs = []
        for i in range(len(self.population)):
            populationIDs.append(i)
        subset = random.sample(populationIDs, tournamentSize)
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
            selectedPaths = self.randomPathSelection(individualID, self.pathSelectionSize)
        # Else rank paths as a set of tuples based on self.pathRankingMethod
        else:
            individual = self.population[individualID]
            rankedPaths = []
            # Compute paths and resize return selection length if necessary
            if len(individual.paths) == 0:
                individual.allUsedPaths()
                # If the number of paths is still zero, return an empty list
                if len(individual.paths) == 0:
                    return []
            thisSelectionSize = self.pathSelectionSize
            if self.pathSelectionSize > len(individual.paths):
                thisSelectionSize = len(individual.paths)
            # Rank paths based on self.pathRankingMethod hyperparameter
            if self.pathRankingMethod == "mostDense":
                for path in individual.paths:
                    rankedPaths.append((path, path.flowPerCostDensity))
                rankedPaths.sort(key=lambda p: p[1], reverse=True)
            elif self.pathRankingMethod == "leastDense":
                for path in individual.paths:
                    rankedPaths.append((path, path.flowPerCostDensity))
                rankedPaths.sort(key=lambda p: p[1], reverse=False)
            elif self.pathRankingMethod == "mostFlow":
                for path in individual.paths:
                    rankedPaths.append((path, path.flow))
                rankedPaths.sort(key=lambda p: p[1], reverse=True)
            elif self.pathRankingMethod == "leastFlow":
                for path in individual.paths:
                    rankedPaths.append((path, path.flow))
                rankedPaths.sort(key=lambda p: p[1], reverse=False)
            elif self.pathRankingMethod == "mostCost":
                for path in individual.paths:
                    rankedPaths.append((path, path.routingCost))
                rankedPaths.sort(key=lambda p: p[1], reverse=True)
            elif self.pathRankingMethod == "leastCost":
                for path in individual.paths:
                    rankedPaths.append((path, path.routingCost))
                rankedPaths.sort(key=lambda p: p[1], reverse=False)
            elif self.pathRankingMethod == "mostEdges":
                for path in individual.paths:
                    rankedPaths.append((path, len(path.edges)))
                rankedPaths.sort(key=lambda p: p[1], reverse=True)
            elif self.pathRankingMethod == "leastEdges":
                for path in individual.paths:
                    rankedPaths.append((path, len(path.edges)))
                rankedPaths.sort(key=lambda p: p[1], reverse=False)
            # Generate selected paths with method call based on pathSelectionMethod
            if self.pathSelectionMethod == "top":
                selectedPaths = self.topPathSelection(rankedPaths, thisSelectionSize)
            elif self.pathSelectionMethod == "roulette":
                selectedPaths = self.rouletteWheelPathSelection(rankedPaths, thisSelectionSize)
            elif self.pathSelectionMethod == "tournament":
                selectedPaths = self.tournamentPathSelection(rankedPaths, thisSelectionSize,
                                                             self.pathTournamentSize)
        return selectedPaths

    def randomPathSelection(self, individualID: int, selectionSize: int) -> list:
        """Returns a random subset of paths in an individual (w/o replacement)"""
        random.seed()
        individual = self.population[individualID]
        # Compute paths and resize return selection length if necessary
        if len(individual.paths) == 0:
            individual.allUsedPaths()
            # If the number of paths is still zero, return an empty list
            if len(individual.paths) == 0:
                return []
        thisSelectionSize = selectionSize
        if selectionSize > len(individual.paths):
            thisSelectionSize = len(individual.paths)
        # Randomly sample the paths list
        selectedPaths = random.sample(individual.paths, thisSelectionSize)
        return selectedPaths

    @staticmethod
    def topPathSelection(rankedPaths: list, selectionSize: int) -> list:
        """Returns the n top paths for the individual based on the path ranking method"""
        # Select paths subset from beginning of full path list
        selectedPaths = []
        for path in range(selectionSize):
            selectedPaths.append(rankedPaths[path][0])
        return selectedPaths

    @staticmethod
    def rouletteWheelPathSelection(rankedPaths: list, selectionSize: int) -> list:
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
        while len(selectedPaths) < selectionSize:
            rng = random.random()
            for p in range(len(rankedPaths)):
                if rng < cumulativeProbabilities[p]:
                    # Utilize a hashable combo of the path's source and sink to prevent duplication
                    hashableID = rankedPaths[p][0].start + rankedPaths[p][0].end
                    if hashableID not in duplicateCheckingSet:
                        selectedPaths.append(rankedPaths[p][0])
                        duplicateCheckingSet.add(hashableID)
                        break
        return selectedPaths

    def tournamentPathSelection(self, rankedPaths: list, selectionSize: int, tournamentSize: int) -> list:
        """Selects the best k paths out of a randomly chosen subset of size n"""
        random.seed()
        # Select random subset of paths
        tournament = random.sample(rankedPaths, tournamentSize)
        # Sort the tournament based on most or least direction
        if self.pathRankingMethod[0] == "m":
            tournament.sort(key=lambda p: p[1], reverse=True)
        elif self.pathRankingMethod[0] == "l":
            rankedPaths.sort(key=lambda p: p[1], reverse=False)
        # Select top and return
        selection = []
        for i in range(selectionSize):
            pathTuple = tournament.pop(0)
            selection.append(pathTuple[0])
        return selection

    # ==============================================
    # ============== MUTATION METHODS ==============
    # ==============================================
    def mutate(self, individualID: int, selectedPaths=None) -> None:
        """Hyper-selection operator that calls specific mutation method based on hyperparameters"""
        if selectedPaths is None:
            selectedPaths = []
        if self.mutationMethod == "randomSingle":
            self.randomSingleMutation(individualID)
        elif self.mutationMethod == "randomTotal":
            self.hypermutateIndividual(individualID)
        elif self.mutationMethod == "pathBased":
            self.selectedPathsMutation(individualID, selectedPaths)

    def selectedPathsMutation(self, individualNum: int, selectedPaths: list) -> None:
        """Mutates all the edges in the selected paths of an individual"""
        individual = self.population[individualNum]
        for path in selectedPaths:
            for edge in path.edges:
                pass

    # =================================================
    # ============== CROSSOVER OPERATORS ==============
    # =================================================
    def crossover(self, parentOneID: int, parentTwoID: int, parentOnePaths=None, parentTwoPaths=None) -> None:
        """Hyper-selection operator that calls specific crossover method based on hyperparameters"""
        random.seed()
        if parentTwoPaths is None:
            parentTwoPaths = []
        if parentOnePaths is None:
            parentOnePaths = []
        if self.selectionMethod == "pathBased":
            self.pathBasedCrossover(parentOneID, parentTwoID, parentOnePaths, parentTwoPaths,
                                    self.replacementStrategy)
        elif self.selectionMethod == "twoPoint":
            self.randomTwoPointCrossover(parentOneID, parentTwoID, self.replacementStrategy)
        elif self.selectionMethod == "onePoint":
            rng = random.random()
            if rng < 0.50:
                self.randomOnePointCrossover(parentOneID, parentTwoID, "fromLeft", self.replacementStrategy)
            else:
                self.randomOnePointCrossover(parentOneID, parentTwoID, "fromRight", self.replacementStrategy)

    def pathBasedCrossover(self, parentOneID: int, parentTwoID: int, parentOnePaths: list,
                           parentTwoPaths: list, replacementStrategy: str) -> None:
        """Crossover based on the flow per cost density of paths of the parents\n
        :param int parentOneID: Index of first parent in population
        :param int parentTwoID: Index of second parent in population
        :param list parentOnePaths: List of paths to be crossed-over from parent 1 to parent 2 (Can be any length)
        :param list parentTwoPaths: List of paths to be crossed-over from parent 2 to parent 1 (Can be any length)
        :param str replacementStrategy: "replaceParents" kills parents; "replaceWeakestTwo" kills two most expensive individuals"""
        # Create two offspring, each identical to one parent
        parentOne = self.population[parentOneID]
        parentTwo = self.population[parentTwoID]
        offspringOne = Individual(self.network)
        offspringOne.alphaValues = copy.deepcopy(parentOne.alphaValues)
        offspringTwo = Individual(self.network)
        offspringTwo.alphaValues = copy.deepcopy(parentTwo.alphaValues)
        # For each path, push all of parent one's alpha values to offspring two
        for path in parentOnePaths:
            for edge in path.edges:
                edgeNum = int(edge.lstrip("e"))
                offspringTwo.alphaValues[edgeNum] = parentOne.alphaValues[edgeNum]
        # For each path, push all of parent one's alpha values to offspring two
        for path in parentTwoPaths:
            for edge in path.edges:
                edgeNum = int(edge.lstrip("e"))
                offspringOne.alphaValues[edgeNum] = parentTwo.alphaValues[edgeNum]
        # Add offspring into the population via the replacement strategy
        if replacementStrategy == "replaceParents":
            self.population[parentOneID] = offspringOne
            self.population[parentTwoID] = offspringTwo
        elif replacementStrategy == "replaceWeakestTwo":
            # Kill weakest two individuals
            self.rankPopulation()
            self.population.pop(-1)
            self.population.pop(-1)
            self.population.append(offspringOne)
            self.population.append(offspringTwo)

    def randomOnePointCrossover(self, parentOneID: int, parentTwoID: int, direction: str,
                                replacementStrategy: str) -> None:
        """Crossover of 2 chromosomes at a single random point\n
        :param int parentOneID: Index of first parent in population
        :param int parentTwoID: Index of second parent in population
        :param str replacementStrategy: "replaceParents" kills parents; "replaceWeakestTwo" kills two most expensive individuals
        :param str direction: "fromLeft" finds crossover point from left; "fromRight" finds crossover point from right"""
        random.seed()
        parentOneChromosome = copy.copy(self.population[parentOneID].alphaValues)
        parentTwoChromosome = copy.copy(self.population[parentTwoID].alphaValues)
        # If from right, reverse input chromosomes
        if direction == "fromRight":
            parentOneChromosome.reverse()
            parentTwoChromosome.reverse()
        crossoverPoint = random.randint(0, self.network.numEdges - 1)
        parentOneLeftGenes = []
        parentTwoLeftGenes = []
        for i in range(crossoverPoint + 1):
            parentOneLeftGenes.append(parentOneChromosome.pop(0))
            parentTwoLeftGenes.append(parentTwoChromosome.pop(0))
        parentOneRightGenes = parentOneChromosome
        parentTwoRightGenes = parentTwoChromosome
        offspringOne = Individual(self.network)
        offspringOne.alphaValues = parentTwoLeftGenes
        for gene in parentOneRightGenes:
            offspringOne.alphaValues.append(gene)
        offspringTwo = Individual(self.network)
        offspringTwo.alphaValues = parentOneLeftGenes
        for gene in parentTwoRightGenes:
            offspringTwo.alphaValues.append(gene)
        # If from right, reverse output chromosomes
        if direction == "fromRight":
            offspringOne.alphaValues.reverse()
            offspringTwo.alphaValues.reverse()
        # Add offspring into the population via the replacement strategy
        if replacementStrategy == "replaceParents":
            self.population[parentOneID] = offspringOne
            self.population[parentTwoID] = offspringTwo
        elif replacementStrategy == "replaceWeakestTwo":
            # Kill weakest two individuals
            self.rankPopulation()
            self.population.pop(-1)
            self.population.pop(-1)
            self.population.append(offspringOne)
            self.population.append(offspringTwo)

    def randomTwoPointCrossover(self, parentOneID: int, parentTwoID: int, replacementStrategy: str) -> None:
        """Crossover of 2 chromosomes at a two random points\n
        :param int parentOneID: Index of first parent in population
        :param int parentTwoID: Index of second parent in population
        :param str replacementStrategy: "replaceParents" kills parents; "replaceWeakestTwo" kills two most expensive individuals"""
        random.seed()
        parentOneChromosome = self.population[parentOneID].alphaValues
        parentTwoChromosome = self.population[parentTwoID].alphaValues
        # If from right, reverse input chromosomes
        crossoverPointOne = random.randint(0, self.network.numEdges - 3)
        crossoverPointTwo = random.randint(crossoverPointOne, self.network.numEdges - 1)
        parentOneInteriorGenes = []
        parentTwoInteriorGenes = []
        for i in range(crossoverPointOne, crossoverPointTwo):
            parentOneInteriorGenes.append(parentOneChromosome[i])
            parentTwoInteriorGenes.append(parentTwoChromosome[i])
        offspringOne = Individual(self.network)
        offspringOne.alphaValues = parentOneChromosome
        offspringTwo = Individual(self.network)
        offspringTwo.alphaValues = parentTwoChromosome
        for i in range(crossoverPointOne, crossoverPointTwo):
            offspringOne.alphaValues[i] = parentTwoInteriorGenes[i - crossoverPointOne]
            offspringTwo.alphaValues[i] = parentOneInteriorGenes[i - crossoverPointOne]
        # Add offspring into the population via the replacement strategy
        if replacementStrategy == "replaceParents":
            self.population[parentOneID] = offspringOne
            self.population[parentTwoID] = offspringTwo
        elif replacementStrategy == "replaceWeakestTwo":
            # Kill weakest two individuals
            self.rankPopulation()
            self.population.pop(-1)
            self.population.pop(-1)
            self.population.append(offspringOne)
            self.population.append(offspringTwo)


