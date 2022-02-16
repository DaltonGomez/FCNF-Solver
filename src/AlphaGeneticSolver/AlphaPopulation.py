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
        # Initialize population
        for i in range(populationSize):
            thisIndividual = AlphaIndividual(self.FCFN)
            thisIndividual.initializeAlphaValuesRandomly()
            self.population.append(thisIndividual)

    def setHyperparameters(self, crossoverRate: float, mutationRate: float):
        """Sets the hyperparameters dictating how the population evolves"""
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate

    # ============================================
    # ============== EVOLUTION LOOP ==============
    # ============================================
    def evolvePopulation(self):
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
        for generation in range(self.numGenerations):

            # CROSSOVER
            self.randomOnePointCrossoverWithoutDeath(0, 1, "fromLeft")
            if random.random() < self.crossoverRate:
                randParentOne = random.randint(0, self.populationSize - 1)
                randParentTwo = random.randint(0, self.populationSize - 1)
                self.randomOnePointCrossoverWithDeath(randParentOne, randParentTwo, "fromLeft")

            # MUTATION
            for individual in range(self.populationSize):
                if random.random() < self.mutationRate * 5:
                    self.randomSingleMutation(individual)
                elif random.random() < self.mutationRate:
                    self.randomTotalMutation(individual)

            # EVALUATE
            self.solvePopulation()
            self.rankPopulation()
            self.visualizeTopIndividual(generation)

    # =================================================
    # ============== SELECTION OPERATORS ==============
    # =================================================
    def topTwoSelection(self):
        """Returns the top two individuals in the population"""
        return [self.population[0], self.population[1]]

    def rouletteWheelSelection(self):
        """Selects individuals probabilistically by their normalized fitness"""
        # TODO - Implement
        pass

    def tournamentSelection(self, tournamentSize: int):
        """Selects the best two individuals out of a randomly chosen subset of size n"""
        # TODO - Implement
        pass

    # =================================================
    # ============== CROSSOVER OPERATORS ==============
    # =================================================
    def randomOnePointCrossoverWithoutDeath(self, parentOneIndex: int, parentTwoIndex: int, direction: str):
        """Crossover of 2 chromosomes at a random point where the bottom 2 individuals die off"""
        random.seed()
        # Kill off the two weakest individuals
        self.rankPopulation()
        self.population.pop(-1)
        self.population.pop(-1)
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
        self.population.append(offspringOne)
        self.population.append(offspringTwo)

    def randomOnePointCrossoverWithDeath(self, parentOneIndex: int, parentTwoIndex: int, direction: str):
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

    def pathBasedCrossover(self):
        """Crossover based on the cost/flow density of all paths"""
        # TODO - Implement

    # =================================================
    # ============== MUTATION OPERATORS ==============
    # =================================================
    def randomSingleMutation(self, individualNum: int):
        """Mutates an individual at a random gene in the chromosome"""
        random.seed()
        mutatePoint = random.randint(0, self.FCFN.numEdges - 1)
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.alphaValues = self.population[individualNum].alphaValues
        mutatedIndividual.alphaValues[mutatePoint] = random.random()
        self.population[individualNum] = mutatedIndividual

    def randomTotalMutation(self, individualNum: int):
        """Mutates the entire chromosome of an individual"""
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.initializeAlphaValuesRandomly()
        self.population[individualNum] = mutatedIndividual

    def randomPathBasedMutation(self, individualNum: int):
        """Mutates one entire path of an individual randomly"""
        random.seed()
        parentIndividual = self.population[individualNum]
        # If the paths have not been computed for the individual, do so
        if len(parentIndividual.paths) == 0:
            parentIndividual.allUsedPaths()
        mutatedPath = random.choice(parentIndividual.paths)
        print(mutatedPath.edges)
        mutatedIndividual = AlphaIndividual(self.FCFN)
        mutatedIndividual.alphaValues = parentIndividual.alphaValues
        for edge in mutatedPath.edges:
            edgeNum = int(edge.lstrip("e"))
            mutatedIndividual.alphaValues[edgeNum] = random.random()  # TODO - Currently randomizing on [0, 1]. Change?
        self.population[individualNum] = mutatedIndividual

    # ============================================
    # ============== HELPER METHODS ==============
    # ============================================
    def solvePopulation(self):
        """Solves the alpha-relaxed LP of each individual in the population"""
        for individual in self.population:
            if individual.isSolved is False:
                individual.executeAlphaSolver(self.minTargetFlow)

    def rankPopulation(self):
        """Ranks the population from least cost to greatest (i.e. fitness)"""
        self.population.sort(key=lambda x: x.trueCost, reverse=False)  # reverse=False ranks least to greatest

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def visualizeTopIndividual(self, generation: int):
        """Prints the data and draws the graph of the top individual"""
        # Print statement & visualization w/ timeout to ensure correct visualization rendering order
        print("Generation= " + str(generation) + "\tBest Individual= " + str(self.population[0].trueCost))
        self.visualizeIndividual(str(generation), 0)  # Second param = 0 --> Top individual
        time.sleep(0.5)

    def visualizeIndividual(self, generation: str, individualRank: int):
        """Draws the .html file for the top individual"""
        self.rankPopulation()
        self.population[individualRank].visualizeAlphaNetwork(endCatName=generation)
