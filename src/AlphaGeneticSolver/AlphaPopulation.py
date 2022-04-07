import copy
import random
import time

from src.AlphaGeneticSolver.AlphaIndividual import AlphaIndividual
from src.AlphaGeneticSolver.AlphaSolver import AlphaSolver
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class AlphaPopulation:
    """Class that manages a population of alpha-relaxed individuals and handles the genetic algorithm operators"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, FCFN: FixedChargeFlowNetwork, minTargetFlow: int, populationSize: int, numGenerations: int):
        """Constructor of a Population instance"""
        # Input network and topology data (NOTE: Input network must be unsolved. If it's not, reload from disc.)
        if FCFN.isSolved is False:
            self.FCFN = FCFN
        else:
            unsolvedFCFN = FixedChargeFlowNetwork()
            unsolvedFCFN.loadFCFN(FCFN.name)
            self.FCFN = unsolvedFCFN
        # Population & Solver Instances
        self.population = []
        self.minTargetFlow = minTargetFlow
        self.alphaSolver = AlphaSolver(self.FCFN, minTargetFlow)

        # GA HYPERPARAMETERS
        # -----------------------
        # Globals HPs
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.alphaBounds = [0.0, 1.0]
        # Individual Selection HPs
        self.selectionMethod = "tournament"  # :param : "tournament", "roulette", "top", "random"
        self.selectionSize = 2
        self.tournamentSize = 2
        # Path Selection HPs
        self.pathSelectionMethod = "roulette"  # :param : "tournament", "roulette", "top", "random"
        self.pathRankingMethod = "leastDense"  # :param : "mostDense", "leastDense", "mostFlow", "leastFlow", "mostCost", "leastCost", "mostEdges", "leastEdges"
        self.pathSelectionSize = 4
        self.pathTournamentSize = 4
        # Crossover HPs
        self.crossoverMethod = "pathBased"  # :param : "onePoint", "twoPoint", "pathBased"
        self.crossoverRate = 0.90
        self.crossoverAttemptsPerGeneration = 1
        self.replacementStrategy = "replaceWeakestTwo"  # : param : "replaceWeakestTwo", "replaceParents"
        # Mutation HPs
        self.mutationMethod = "pathBased"  # :param : "randomSingle", "randomTotal", "pathBased"
        self.mutationRate = 0.50

    # ==================================================================
    # ============== HYPERPARAMETER SETTERS & INITIALIZER ==============
    # ==================================================================
    def setAlphaBounds(self, lowerBound: float, upperBound: float) -> None:
        """Sets the GA class field that dictates the range when randomly initializing/updating alpha values"""
        self.alphaBounds = [lowerBound, upperBound]

    def setIndividualSelectionHyperparams(self, selectionMethod: str, selectionSize: int, tournamentSize: int) -> None:
        """Sets the GA class fields that dictate how the selection of individuals is carried out \n
        :param str selectionMethod: One of following: {"tournament", "roulette", "top", "random"}
        :param int selectionSize: Number of individuals returned (NOTE: Currently overridden to always be 2!)
        :param int tournamentSize: Size of tournament subset selected if selectionMethod = "tournament"
        """
        self.selectionMethod = selectionMethod
        self.selectionSize = 2  # NOTE: Individual selection is for crossover, which always takes only two individuals
        # self.selectionSize = selectionSize
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

    def initializePopulation(self, initialAlphas: list) -> None:
        """Initializes the population with alpha values, solves each individual, and ranks"""
        # Initialize population
        for i in range(self.populationSize):
            thisIndividual = AlphaIndividual(self.FCFN)
            thisIndividual.initializeAlphaValuesRandomly(lowerBound=self.alphaBounds[0], upperBound=self.alphaBounds[1])
            self.population.append(thisIndividual)
        self.solvePopulation()
        self.rankPopulation()

    # ============================================
    # ============== EVOLUTION LOOP ==============
    # ============================================
    def evolvePopulation(self, drawing=True) -> AlphaIndividual:
        """Evolves the population based on the selection, crossover and mutation operators\n
        :param bool drawing: A boolean to enable/disable top individual html rendering per generation"""
        random.seed()
        # INITIALIZATION
        if len(self.population) == 0:
            self.initializePopulation([0.0, 1.0])
        # MAIN EVOLUTION LOOP
        for generation in range(self.numGenerations):
            # ATTEMPT CROSSOVER
            for crossoverAttempt in range(self.crossoverAttemptsPerGeneration):
                if random.random() < self.crossoverRate:
                    # SELECT INDIVIDUALS
                    selectedIndividuals = self.selectIndividuals()
                    parentOnePaths = self.selectPaths(selectedIndividuals[0])
                    parentTwoPaths = self.selectPaths(selectedIndividuals[1])
                    # CROSSOVER
                    self.crossover(selectedIndividuals[0], selectedIndividuals[1], parentOnePaths, parentTwoPaths)

            # MUTATION
            for individual in range(len(self.population)):
                if random.random() < self.mutationRate:
                    selectedPaths = self.selectPaths(individual)
                    self.mutate(individual, selectedPaths)

            # EVALUATE
            self.solvePopulation()
            self.rankPopulation()
            self.printAllCosts(generation)
            if drawing is True:
                self.visualizeIndividual(generation, 0, graphType="withLabels")

                # TODO - Delete prints to console; needed screenshots for presentation
                bestIndividual = self.population[0]
                print("Best Alpha Values:")
                print(bestIndividual.alphaValues)
                flowVector = []
                for edge in bestIndividual.FCFN.edgesDict:
                    if edge in bestIndividual.openedEdgesDict:
                        flowVector.append(bestIndividual.openedEdgesDict[edge][0])
                    else:
                        flowVector.append(0)
                    # print(edge)
                print("Best Flow Values:")
                print(flowVector)
                # TODO - Delete prints to console; needed screenshots for presentation
        return self.population[0]

    # ======================================================
    # ============== SOLVER & RANKING METHODS ==============
    # ======================================================
    def solvePopulation(self) -> None:
        """Solves the alpha-relaxed LP of each individual in the population"""
        for individual in self.population:
            if individual.isSolved is False:
                self.solveIndividual(individual)

    def solveIndividual(self, individual: AlphaIndividual) -> None:
        """Solves a single individual by calling AlphaSolver and returns the solution to the individual"""
        relaxedCoefficients = individual.computeRelaxedCoefficients()
        self.alphaSolver.updateObjectiveFunction(relaxedCoefficients)
        self.alphaSolver.solveModel()
        self.alphaSolver.writeSolution(individual)
        individual.calculateTrueCost()

    def rankPopulation(self) -> None:
        """Ranks the population from least cost to greatest (i.e. fitness)"""
        self.population.sort(key=lambda x: x.trueCost, reverse=False)  # reverse=False ranks least to greatest

    # ============================================================
    # ============== INDIVIDUAL SELECTION OPERATORS ==============
    # ============================================================
    def selectIndividuals(self) -> list:
        """Hyper-selection operator that calls specific selection method based on hyperparameters"""
        selectedIndividualIDs = []
        if self.selectionMethod == "tournament":
            selectedIndividualIDs = self.tournamentSelection(self.selectionSize, self.tournamentSize)
        elif self.selectionMethod == "roulette":
            selectedIndividualIDs = self.rouletteWheelSelection(self.selectionSize)
        elif self.selectionMethod == "top":
            selectedIndividualIDs = self.topSelection(self.selectionSize)
        elif self.selectionMethod == "random":
            selectedIndividualIDs = self.randomSelection(self.selectionSize)
        return selectedIndividualIDs

    def randomSelection(self, selectionSize: int) -> list:
        """Returns a random subset of individuals in the population (w/o replacement)"""
        random.seed()
        populationIDs = []
        for i in range(len(self.population)):
            populationIDs.append(i)
        randomIndividualsIDs = random.sample(populationIDs, selectionSize)
        return randomIndividualsIDs

    def topSelection(self, selectionSize: int) -> list:
        """Returns the top n individuals in the population"""
        self.rankPopulation()
        topIndividualIDs = []
        # topIndividuals = []
        for i in range(selectionSize):
            topIndividualIDs.append(i)
            # topIndividuals.append(self.population[i])
        return topIndividualIDs

    def rouletteWheelSelection(self, selectionSize: int) -> list:
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
            cumulativeProbabilities.append((fitnessFromCost[i] / cumulativeFitness) + cumulativeProbabilities[i - 1])
        selectionSet = set()
        while len(selectionSet) < selectionSize:
            rng = random.random()
            for individual in range(len(self.population)):
                if rng < cumulativeProbabilities[individual]:
                    selectionSet.add(individual)
                    break
        return list(selectionSet)

    def tournamentSelection(self, selectionSize: int, tournamentSize: int) -> list:
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
        for i in range(selectionSize):
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
                selectedPaths = self.tournamentPathSelection(rankedPaths, thisSelectionSize, self.pathTournamentSize)
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
        thisSelectionSize = self.pathSelectionSize
        if self.pathSelectionSize > len(individual.paths):
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
            self.pathBasedCrossover(parentOneID, parentTwoID, parentOnePaths, parentTwoPaths, self.replacementStrategy)
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
        random.seed()
        # Create two offspring, each identical to one parent
        parentOne = self.population[parentOneID]
        parentTwo = self.population[parentTwoID]
        offspringOne = AlphaIndividual(self.FCFN)
        offspringOne.alphaValues = copy.deepcopy(parentOne.alphaValues)
        offspringTwo = AlphaIndividual(self.FCFN)
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
        crossoverPoint = random.randint(0, self.FCFN.numEdges - 1)
        parentOneLeftGenes = []
        parentTwoLeftGenes = []
        for i in range(crossoverPoint + 1):
            parentOneLeftGenes.append(parentOneChromosome.pop(0))
            parentTwoLeftGenes.append(parentTwoChromosome.pop(0))
        parentOneRightGenes = parentOneChromosome
        parentTwoRightGenes = parentTwoChromosome
        offspringOne = AlphaIndividual(self.FCFN)
        offspringOne.alphaValues = parentTwoLeftGenes
        for gene in parentOneRightGenes:
            offspringOne.alphaValues.append(gene)
        offspringTwo = AlphaIndividual(self.FCFN)
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
        crossoverPointOne = random.randint(0, self.FCFN.numEdges - 3)
        crossoverPointTwo = random.randint(crossoverPointOne, self.FCFN.numEdges - 1)
        parentOneInteriorGenes = []
        parentTwoInteriorGenes = []
        for i in range(crossoverPointOne, crossoverPointTwo):
            parentOneInteriorGenes.append(parentOneChromosome[i])
            parentTwoInteriorGenes.append(parentTwoChromosome[i])
        offspringOne = AlphaIndividual(self.FCFN)
        offspringOne.alphaValues = parentOneChromosome
        offspringTwo = AlphaIndividual(self.FCFN)
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

    # =================================================
    # ============== MUTATION OPERATORS ==============
    # =================================================
    def mutate(self, individualID: int, selectedPaths=None) -> None:
        """Hyper-selection operator that calls specific mutation method based on hyperparameters"""
        random.seed()
        if selectedPaths is None:
            selectedPaths = []
        if self.mutationMethod == "randomSingle":
            self.randomSingleMutation(individualID)
        elif self.mutationMethod == "randomTotal":
            self.randomTotalMutation(individualID)
        elif self.mutationMethod == "pathBased":
            self.selectedPathsMutation(individualID, selectedPaths)

    def randomSingleMutation(self, individualNum: int) -> None:
        """Mutates an individual at only one random gene in the chromosome"""
        random.seed()
        mutatePoint = random.randint(0, self.FCFN.numEdges - 1)
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.alphaValues = self.population[individualNum].alphaValues
        mutatedIndividual.alphaValues[mutatePoint] = random.uniform(self.alphaBounds[0], self.alphaBounds[1])
        self.population[individualNum] = mutatedIndividual

    def randomTotalMutation(self, individualNum: int) -> None:
        """Mutates the entire chromosome of an individual"""
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.initializeAlphaValuesRandomly(lowerBound=self.alphaBounds[0], upperBound=self.alphaBounds[1])
        self.population[individualNum] = mutatedIndividual

    def selectedPathsMutation(self, individualNum: int, selectedPaths: list) -> None:
        """Mutates all the edges in the selected paths of an individual"""
        random.seed()
        parentIndividual = self.population[individualNum]
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.alphaValues = parentIndividual.alphaValues
        for path in selectedPaths:
            for edge in path.edges:
                edgeNum = int(edge.lstrip("e"))
                mutatedIndividual.alphaValues[edgeNum] = random.uniform(self.alphaBounds[0], self.alphaBounds[1])
        self.population[individualNum] = mutatedIndividual

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def visualizeIndividual(self, generation: int, individualRank: int, graphType="fullGraph") -> None:
        """Draws the .html file for an individual"""
        # Visualization w/ timeout to ensure the correct rendering order
        self.population[individualRank].visualizeAlphaNetwork(endCatName=str(generation), graphType=graphType)
        time.sleep(0.25)

    def printAllCosts(self, generation: int) -> None:
        """Prints an ordered list of the individual's cost"""
        costList = []
        for i in range(len(self.population)):
            costList.append(round(self.population[i].trueCost))
        print("Generation= " + str(generation) + "\tBest Individual= " + str(
            self.population[0].trueCost) + "\tFull Cost List:")
        print(costList)

    def printCurrentSolverOverview(self) -> None:
        """Prints out the current status of the solver"""
        self.alphaSolver.printCurrentSolverDetails()
        # self.alphaSolver.printCurrentModel()
        # self.alphaSolver.printCurrentSolution()
