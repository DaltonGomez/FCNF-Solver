import copy
import random
import time

from src.AlphaGeneticSolver.AlphaIndividual import AlphaIndividual
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
        # Population/GA Attributes
        self.minTargetFlow = minTargetFlow
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.population = []
        # Evolution Hyperparameters- Tune with setHyperparameters() method
        self.crossoverRate = 0.75
        self.mutationRate = 0.05

    def setHyperparameters(self, crossoverRate: float, mutationRate: float) -> None:
        """Sets the hyperparameters dictating how the population evolves"""
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate

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
    def evolvePopulation(self) -> AlphaIndividual:
        """Evolves the population based on the selection, crossover and mutation operators
        Pseudocode:
        initialize(population)
        while termination is not met:
            crossover(selection(population))
            mutate(selection(population))
            evaluate(population)
        return bestIndividual(population)
        """
        random.seed()
        # Ensure population was initialized; otherwise, do so
        if len(self.population) == 0:
            self.initializePopulation([0.0, 1.0])
        # MAIN EVOLUTION LOOP
        for generation in range(self.numGenerations):

            # CROSSOVER
            if random.random() < self.crossoverRate:
                self.randomOnePointCrossoverWithParentReplacement(0, 1, "fromLeft")

            # MUTATION
            for individual in range(len(self.population)):
                if random.random() < self.mutationRate:
                    self.randomSingleMutation(individual)

            # EVALUATE
            self.solvePopulation()
            self.rankPopulation()
            self.visualizeTopIndividual(generation)
        return self.population[0]

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

    def tournamentSelection(self, tournamentSize: int, selectionSize: int) -> list:
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
        if selectionSize > len(individual.paths):
            selectionSize = len(individual.paths)
        # Randomly sample the paths list
        selectedPaths = random.sample(individual.paths, selectionSize)
        return selectedPaths

    def densityBasedPathSelection(self, individualID: int, selectionSize: int, selectionOrder: str) -> list:
        """Returns the n most dense (flow/cost ratio) paths for the individual
        @:param selectionOrder = {"mostDense", "leastDense"}"""
        individual = self.population[individualID]
        # Compute paths and resize return selection length if necessary
        if len(individual.paths) == 0:
            individual.allUsedPaths()
        if selectionSize > len(individual.paths):
            selectionSize = len(individual.paths)
        # Sort based on selectionOrder
        if selectionOrder == "mostDense":
            individual.paths.sort(key=lambda p: p.flowPerCostDensity, reverse=True)
        elif selectionOrder == "leastDense":
            individual.paths.sort(key=lambda p: p.flowPerCostDensity, reverse=False)
        # Select paths subset from beginning of full path list
        selectedPaths = []
        for p in range(selectionSize):
            selectedPaths.append(individual.paths[p])
        return selectedPaths

    def rouletteWheelPathSelection(self, individualID: int, selectionSize: int, selectionOrder: str) -> list:
        """Selects paths from an individual probabilistically by their normalized fitness (i.e. flow per cost density)
        @:param selectionOrder = {"mostDense", "leastDense"}"""
        random.seed()
        individual = self.population[individualID]
        # Compute paths and resize return selection length if necessary
        if len(individual.paths) == 0:
            individual.allUsedPaths()
        if selectionSize > len(individual.paths):
            selectionSize = len(individual.paths)
        # Sort based on selectionOrder
        if selectionOrder == "mostDense":
            individual.paths.sort(key=lambda p: p.flowPerCostDensity, reverse=True)
        elif selectionOrder == "leastDense":
            individual.paths.sort(key=lambda p: p.flowPerCostDensity, reverse=False)
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

    def tournamentPathSelection(self, individualID: int, tournamentSize: int, selectionSize: int,
                                selectionOrder: str) -> list:
        """Selects the best k paths out of a randomly chosen subset of size n
        @:param selectionOrder = {"mostDense", "leastDense"}"""
        random.seed()
        individual = self.population[individualID]
        # Compute paths and resize return selection length if necessary
        if len(individual.paths) == 0:
            individual.allUsedPaths()
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
    def pathBasedCrossoverWithParentReplacement(self, parentOneID: int, parentTwoID: int, parentOnePaths: list,
                                                parentTwoPaths: list, replacementStrategy: str) -> None:
        """Crossover based on the flow per cost density of paths of the parents
        @:param replacementStrategy = {"replaceParents", "replaceWeakestTwo"}"""
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
            for edge in path:
                edgeNum = int(edge.lstrip("e"))
                offspringTwo.alphaValues[edgeNum] = parentOne.alphaValues[edgeNum]
        # For each path, push all of parent one's alpha values to offspring two
        for path in parentTwoPaths:
            for edge in path:
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
        """Crossover of 2 chromosomes at a single random point
        @:param direction = {"fromLeft", "fromRight"}
        @:param replacementStrategy = {"replaceParents", "replaceWeakestTwo"}"""
        random.seed()
        parentOneChromosome = copy.deepcopy(self.population[parentOneID].alphaValues)
        parentTwoChromosome = copy.deepcopy(self.population[parentTwoID].alphaValues)
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

    # ============================================
    # ============== HELPER METHODS ==============
    # ============================================
    def solvePopulation(self) -> None:
        """Solves the alpha-relaxed LP of each individual in the population"""
        for individual in self.population:
            if individual.isSolved is False:
                individual.executeAlphaSolver(self.minTargetFlow)

    def rankPopulation(self) -> None:
        """Ranks the population from least cost to greatest (i.e. fitness)"""
        self.population.sort(key=lambda x: x.trueCost, reverse=False)  # reverse=False ranks least to greatest

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def visualizeTopIndividual(self, generation: int) -> None:
        """Prints the data and draws the graph of the top individual"""
        # Print statement & visualization w/ timeout to ensure correct visualization rendering order
        print("Generation= " + str(generation) + "\tBest Individual= " + str(self.population[0].trueCost))
        self.visualizeIndividual(str(generation), 0)  # Second param = 0 --> Top individual

    def visualizeIndividual(self, generation: str, individualRank: int) -> None:
        """Draws the .html file for an individual"""
        self.rankPopulation()
        self.population[individualRank].visualizeAlphaNetwork(endCatName=generation)
        time.sleep(0.25)

    def printAllCosts(self) -> None:
        """Prints an ordered list of the individual's cost"""
        costList = []
        for i in range(len(self.population)):
            costList.append(round(self.population[i].trueCost))
        print(costList)
