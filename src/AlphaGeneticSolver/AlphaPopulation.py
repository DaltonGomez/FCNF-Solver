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
        self.minTargetFlow = minTargetFlow
        self.alphaSolver = AlphaSolver(self.FCFN, minTargetFlow)
        self.population = []
        # Evolution Hyperparameters- Tune with setHyperparameters() method
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.crossoverRate = 0.90
        self.mutationRate = 0.03

    def setHyperparameters(self, crossoverRate: float, mutationRate: float) -> None:
        """Sets the hyperparameters dictating how the population evolves"""
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        # TODO - Add in additional hyperparameters and build tuning experiment

    def initializePopulation(self, initialAlphas: list) -> None:
        """Initializes the population with alpha values, solves each individual, and ranks"""
        # Initialize population
        for i in range(self.populationSize):
            thisIndividual = AlphaIndividual(self.FCFN)
            thisIndividual.initializeAlphaValuesRandomly(lowerBound=initialAlphas[0], upperBound=initialAlphas[1])
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
        crossoverAttemptsPerGeneration = 1
        for generation in range(self.numGenerations):
            for crossover in range(crossoverAttemptsPerGeneration):
                if random.random() < self.crossoverRate:
                    # SELECTION
                    individuals = self.tournamentSelection(2, 2)
                    individualZeroPaths = self.rouletteWheelPathSelection(individuals[0], 1, "mostDense")
                    individualOnePaths = self.rouletteWheelPathSelection(individuals[1], 1, "mostDense")
                    # CROSSOVER
                    self.pathBasedCrossover(individuals[0], individuals[1], individualZeroPaths, individualOnePaths,
                                            "replaceWeakestTwo")
            # MUTATION
            for individual in range(len(self.population)):
                if random.random() < self.mutationRate:
                    selectedPaths = self.rouletteWheelPathSelection(individual, 1, "mostDense")
                    self.selectedPathsMutation(individual, selectedPaths)
            # EVALUATE
            self.solvePopulation()
            self.rankPopulation()
            self.printAllCosts(generation)
            if drawing is True:
                self.visualizeIndividual(generation, 0)
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
        self.alphaSolver.updateObjectiveFunction(individual.alphaValues)
        self.alphaSolver.solveModel()
        self.alphaSolver.writeSolution(individual)
        individual.calculateTrueCost()

    def rankPopulation(self) -> None:
        """Ranks the population from least cost to greatest (i.e. fitness)"""
        self.population.sort(key=lambda x: x.trueCost, reverse=False)  # reverse=False ranks least to greatest

    # ============================================================
    # ============== INDIVIDUAL SELECTION OPERATORS ==============
    # ============================================================
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
        if selectionSize > len(individual.paths):
            selectionSize = len(individual.paths)
        # Randomly sample the paths list
        selectedPaths = random.sample(individual.paths, selectionSize)
        return selectedPaths

    def densityBasedPathSelection(self, individualID: int, selectionSize: int, selectionOrder: str) -> list:
        """Returns the n most dense (flow/cost ratio) paths for the individual\n
        :param int individualID: Index of individual in population
        :param int selectionSize: Number of paths returned
        :param str selectionOrder: "mostDense" selects with highest flow per cost density; "leastDense" with lowest"""
        individual = self.population[individualID]
        # Compute paths and resize return selection length if necessary
        if len(individual.paths) == 0:
            individual.allUsedPaths()
            # If the number of paths is still zero, return an empty list
            if len(individual.paths) == 0:
                return []
        if selectionSize > len(individual.paths):
            selectionSize = len(individual.paths)
        # Sort based on selectionOrder
        if selectionOrder == "mostDense":
            individual.paths.sort(key=lambda x: x.flowPerCostDensity, reverse=True)
        elif selectionOrder == "leastDense":
            individual.paths.sort(key=lambda x: x.flowPerCostDensity, reverse=False)
        # Select paths subset from beginning of full path list
        selectedPaths = []
        for p in range(selectionSize):
            selectedPaths.append(individual.paths[p])
        return selectedPaths

    def rouletteWheelPathSelection(self, individualID: int, selectionSize: int, selectionOrder: str) -> list:
        """Selects paths from an individual probabilistically by their normalized fitness (i.e. flow per cost density)\n
        :param int individualID: Index of individual in population
        :param int selectionSize: Number of paths returned
        :param str selectionOrder: "mostDense" selects with highest flow per cost density; "leastDense" with lowest"""
        random.seed()
        individual = self.population[individualID]
        # Compute paths and resize return selection length if necessary
        if len(individual.paths) == 0:
            individual.allUsedPaths()
            # If the number of paths is still zero, return an empty list
            if len(individual.paths) == 0:
                return []
        if selectionSize > len(individual.paths):
            selectionSize = len(individual.paths)
        # Sort based on selectionOrder
        if selectionOrder == "mostDense":
            individual.paths.sort(key=lambda x: x.flowPerCostDensity, reverse=True)
        elif selectionOrder == "leastDense":
            individual.paths.sort(key=lambda x: x.flowPerCostDensity, reverse=False)
        # Build cumulative probability function
        cumulativeFitness = 0
        for p in range(len(individual.paths)):
            cumulativeFitness += individual.paths[p].flowPerCostDensity
        cumulativeProbabilities = [individual.paths[0].flowPerCostDensity / cumulativeFitness]
        for i in range(1, len(individual.paths)):
            cumulativeProbabilities.append(
                (individual.paths[i].flowPerCostDensity / cumulativeFitness) + cumulativeProbabilities[i - 1])
        # Build selected paths set
        selectedPaths = []
        duplicateCheckingSet = set()
        while len(selectedPaths) < selectionSize:
            rng = random.random()
            for p in range(len(individual.paths)):
                if rng < cumulativeProbabilities[p]:
                    # Utilize a hashable combo of the path's source and sink to prevent duplication
                    hashableID = individual.paths[p].start + individual.paths[p].end
                    if hashableID not in duplicateCheckingSet:
                        selectedPaths.append(individual.paths[p])
                        duplicateCheckingSet.add(hashableID)
                        break
        return selectedPaths

    def tournamentPathSelection(self, individualID: int, selectionSize: int, tournamentSize: int,
                                selectionOrder: str) -> list:
        """Selects the best k paths out of a randomly chosen subset of size n\n
        :param int individualID: Index of individual in population
        :param int selectionSize: Number of paths returned
        :param int tournamentSize: Size of tournament used to generate selection
        :param str selectionOrder: "mostDense" selects with highest flow per cost density; "leastDense" with lowest"""
        random.seed()
        individual = self.population[individualID]
        # Compute paths and resize return selection length if necessary
        if len(individual.paths) == 0:
            individual.allUsedPaths()
            # If the number of paths is still zero, return an empty list
            if len(individual.paths) == 0:
                return []
        if selectionSize > len(individual.paths):
            selectionSize = len(individual.paths)
        # Select random subset of paths
        tournament = random.sample(individual.paths, tournamentSize)
        # Sort based on selectionOrder
        if selectionOrder == "mostDense":
            tournament.sort(key=lambda p: p.flowPerCostDensity, reverse=True)
        elif selectionOrder == "leastDense":
            tournament.sort(key=lambda p: p.flowPerCostDensity, reverse=False)
        # Select and return
        selection = []
        for i in range(selectionSize):
            selection.append(tournament.pop(0))
        return selection

    # =================================================
    # ============== CROSSOVER OPERATORS ==============
    # =================================================
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
    def randomSingleMutation(self, individualNum: int, lowerBound=0.0, upperBound=1.0) -> None:
        """Mutates an individual at only one random gene in the chromosome"""
        random.seed()
        mutatePoint = random.randint(0, self.FCFN.numEdges - 1)
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.alphaValues = self.population[individualNum].alphaValues
        mutatedIndividual.alphaValues[mutatePoint] = random.uniform(lowerBound, upperBound)
        self.population[individualNum] = mutatedIndividual

    def randomTotalMutation(self, individualNum: int, lowerBound=0.0, upperBound=1.0) -> None:
        """Mutates the entire chromosome of an individual"""
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.initializeAlphaValuesRandomly(lowerBound=lowerBound, upperBound=upperBound)
        self.population[individualNum] = mutatedIndividual

    def selectedPathsMutation(self, individualNum: int, selectedPaths: list, lowerBound=0.0, upperBound=1.0) -> None:
        """Mutates all the edges in the selected paths of an individual"""
        random.seed()
        parentIndividual = self.population[individualNum]
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.alphaValues = parentIndividual.alphaValues
        for path in selectedPaths:
            for edge in path.edges:
                edgeNum = int(edge.lstrip("e"))
                mutatedIndividual.alphaValues[edgeNum] = random.uniform(lowerBound, upperBound)
        self.population[individualNum] = mutatedIndividual

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def visualizeIndividual(self, generation: int, individualRank: int) -> None:
        """Draws the .html file for an individual"""
        # Visualization w/ timeout to ensure the correct rendering order
        self.population[individualRank].visualizeAlphaNetwork(endCatName=str(generation))
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
