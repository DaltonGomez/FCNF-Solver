import random
import time

from src.AlphaGeneticSolver.AlphaIndividual import AlphaIndividual
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class AlphaPopulation:
    """Class that manages a population of alpha-reduced individuals and handles the genetic algorithm operators"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, FCFNinstance: FixedChargeFlowNetwork, minTargetFlow: int, populationSize: int,
                 numGenerations: int):
        """Constructor of a Population instance"""
        # Input network and topology data (NOTE: Input network must be unsolved. If it's not, reload from disc.)
        if FCFNinstance.isSolved is False:
            self.FCFN = FCFNinstance
        else:
            unsolvedFCFN = FixedChargeFlowNetwork()
            unsolvedFCFN.loadFCFN(FCFNinstance.name)
            self.FCFN = unsolvedFCFN

        # Population/GA attributes
        self.minTargetFlow = minTargetFlow
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.population = []

        # Initialize population
        for i in range(populationSize):
            thisIndividual = AlphaIndividual(self.FCFN)
            thisIndividual.initializeAlphaValues("random")
            self.population.append(thisIndividual)

    # ============================================
    # ============== EVOLUTION LOOP ==============
    # ============================================
    def evolvePopulation(self, softMutationRate: float, hardMutationRate: float, crossoverRate: float):
        """Evolves the population based on the crossover and mutation operators"""
        random.seed()
        for generation in range(self.numGenerations):
            # Solve unsolved instances, re-rank, and display top individual
            self.solvePopulation()
            self.rankPopulation()
            self.visualizeTopIndividual(generation)

            # Crossover Operators
            if random.random() < crossoverRate:
                parentOne = random.randint(0, self.populationSize - 1)
                parentTwo = random.randint(0, self.populationSize - 1)
                self.randomSinglePointCrossover(parentOne, parentTwo)

            # Mutation operators
            for individual in range(self.populationSize):
                if random.random() < softMutationRate:
                    self.randomSingleMutation(individual)
                elif random.random() < hardMutationRate:
                    self.randomTotalMutation(individual)

    # =================================================
    # ============== CROSSOVER OPERATORS ==============
    # =================================================
    def randomSinglePointCrossover(self, parentOneIndex: int, parentTwoIndex: int):
        """Crossover of two individual's alpha chromosome at a random point"""
        # TODO - Implement from the right side (as well as the left which is what this does)
        random.seed()
        parentOne = self.population.pop(parentOneIndex)
        if parentOneIndex < parentTwoIndex:
            parentTwo = self.population.pop(parentTwoIndex - 1)  # Adjust for parent two's index change after pop
        else:
            parentTwo = self.population.pop(parentTwoIndex)
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

    # ============================================
    # ============== HELPER METHODS ==============
    # ============================================
    def solvePopulation(self):
        """Solves the alpha-reduced LP of each individual in the population"""
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
        self.population[individualRank].visualizeAlphaNetwork(frontCatName=generation + "-")
