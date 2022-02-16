import copy
import random
import sys
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

    def initializePopulation(self, initialAlphaRange: list) -> None:
        """Initializes the population with alpha values, solves each individual, and ranks"""
        # Initialize population
        for i in range(self.populationSize):
            thisIndividual = AlphaIndividual(self.FCFN)
            thisIndividual.initializeAlphaValuesRandomly(lowerBound=initialAlphaRange[0],
                                                         upperBound=initialAlphaRange[1])
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
            for individual in range(self.populationSize):
                if random.random() < self.mutationRate:
                    self.randomSingleMutation(individual)

            # EVALUATE
            self.solvePopulation()
            self.rankPopulation()
            self.visualizeTopIndividual(generation)
        return self.population[0]

    # =================================================
    # ============== SELECTION OPERATORS ==============
    # =================================================
    def randomSelection(self, selectionSize: int) -> list:
        """Returns a random subset of individuals in the population (w/o replacement)"""
        random.seed()
        populationIDs = []
        for i in range(self.populationSize):
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

    def rouletteWheelSelection(self) -> list:
        """Selects individuals probabilistically by their normalized fitness"""
        self.rankPopulation()

        pass

    def tournamentSelection(self, tournamentSize: int, selectionSize: int) -> list:
        """Selects the best two individuals out of a randomly chosen subset of size n"""
        random.seed()
        # Select subset of population
        populationIDs = []
        for i in range(self.populationSize):
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

    # =================================================
    # ============== CROSSOVER OPERATORS ==============
    # =================================================
    def randomOnePointCrossoverWithWeakestTwoReplacement(self, parentOneIndex: int, parentTwoIndex: int,
                                                         direction: str) -> None:
        """Crossover of 2 chromosomes at a random point where the bottom 2 individuals die off"""
        random.seed()
        self.rankPopulation()
        parentOneChromosome = copy.deepcopy(self.population[parentOneIndex].alphaValues)
        parentTwoChromosome = copy.deepcopy(self.population[parentTwoIndex].alphaValues)
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
        # Kill off weakest two individuals
        self.population.pop(-1)
        self.population.pop(-1)
        # Add in offspring
        self.population.append(offspringOne)
        self.population.append(offspringTwo)

    def randomOnePointCrossoverWithParentReplacement(self, parentOneIndex: int, parentTwoIndex: int,
                                                     direction: str) -> None:
        """Crossover of 2 individual's chromosome at a random point where the parents are removed from the population"""
        random.seed()
        parentOne = self.population.pop(parentOneIndex)
        if parentOneIndex < parentTwoIndex:
            parentTwo = self.population.pop(parentTwoIndex - 1)  # Adjust for parent two's index change after pop
        else:
            parentTwo = self.population.pop(parentTwoIndex)
        if direction == "fromRight":
            parentOne.alphaValues.reverse()
            parentTwo.alphaValues.reverse()
        crossoverPoint = random.randint(0, self.FCFN.numEdges - 1)
        parentOneLeftGenes = []
        parentTwoLeftGenes = []
        for i in range(crossoverPoint + 1):
            parentOneLeftGenes.append(parentOne.alphaValues.pop(0))
            parentTwoLeftGenes.append(parentTwo.alphaValues.pop(0))
        parentOneRightGenes = parentOne.alphaValues
        parentTwoRightGenes = parentTwo.alphaValues
        offspringOne = AlphaIndividual(self.FCFN)
        offspringOne.alphaValues = parentTwoLeftGenes
        for gene in parentOneRightGenes:
            offspringOne.alphaValues.append(gene)
        offspringTwo = AlphaIndividual(self.FCFN)
        offspringTwo.alphaValues = parentOneLeftGenes
        for gene in parentTwoRightGenes:
            offspringTwo.alphaValues.append(gene)
        self.population.append(offspringOne)
        self.population.append(offspringTwo)

    def pathBasedCrossover(self) -> None:
        """Crossover based on the cost/flow density of all paths"""
        # TODO - Implement

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

    def randomSinglePathMutation(self, individualNum: int, lowerBound=0.0, upperBound=1.0) -> None:
        """Mutates one entire path of an individual randomly"""
        random.seed()
        parentIndividual = self.population[individualNum]
        # If the paths have not been computed for the individual, do so
        if len(parentIndividual.paths) == 0:
            parentIndividual.allUsedPaths()
        mutatedPath = random.choice(parentIndividual.paths)
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.alphaValues = parentIndividual.alphaValues
        for edge in mutatedPath.edges:
            edgeNum = int(edge.lstrip("e"))
            mutatedIndividual.alphaValues[edgeNum] = random.uniform(lowerBound, upperBound)
        self.population[individualNum] = mutatedIndividual

    def mostDensePathMutation(self, individualNum: int, lowerBound=0.0, upperBound=1.0) -> None:
        """Mutates an individual's entire path that has the highest cost/flow density"""
        parentIndividual = self.population[individualNum]
        # If the paths have not been computed for the individual, do so
        if len(parentIndividual.paths) == 0:
            parentIndividual.allUsedPaths()
            parentIndividual.printAllPathData()
        maxDensity = 0
        mutatedPath = None
        for path in parentIndividual.paths:
            if path.totalCostPerFlow > maxDensity:
                maxDensity = path.totalCostPerFlow
                mutatedPath = path
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.alphaValues = parentIndividual.alphaValues
        for edge in mutatedPath.edges:
            edgeNum = int(edge.lstrip("e"))
            mutatedIndividual.alphaValues[edgeNum] = random.uniform(lowerBound, upperBound)
        self.population[individualNum] = mutatedIndividual

    def leastDensePathMutation(self, individualNum: int, lowerBound=0.0, upperBound=1.0) -> None:
        """Mutates an individual's entire path that has the lowest cost/flow density"""
        parentIndividual = self.population[individualNum]
        # If the paths have not been computed for the individual, do so
        if len(parentIndividual.paths) == 0:
            parentIndividual.allUsedPaths()
            parentIndividual.printAllPathData()
        minDensity = sys.maxsize
        mutatedPath = None
        for path in parentIndividual.paths:
            if path.totalCostPerFlow < minDensity:
                minDensity = path.totalCostPerFlow
                mutatedPath = path
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.alphaValues = parentIndividual.alphaValues
        for edge in mutatedPath.edges:
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
        """Draws the .html file for the top individual"""
        self.rankPopulation()
        self.population[individualRank].visualizeAlphaNetwork(endCatName=generation)
        time.sleep(0.25)
