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
    def evolvePopulation(self, softMutationRate: float, hardMutationRate: float):
        """Evolves the population based on the crossover and mutation operators"""
        random.seed()
        for generation in range(self.numGenerations):
            # Solve unsolved instances and re-rank
            self.solvePopulation()
            self.rankPopulation()

            # Print statement & visualization w/ timeout to ensure correct visualization rendering order
            print("Generation= " + str(generation) + "\tBest Individual= " + str(self.population[0].trueCost))
            self.visualizeIndividual(str(generation), 0)  # Second param = 0 --> Top individual
            time.sleep(0.5)

            # Genetic operators
            for individual in range(self.populationSize):
                if random.random() < softMutationRate:
                    self.randomSinglePointMutation(individual)
                elif random.random() < hardMutationRate:
                    self.randomTotalMutation(individual)

    # =================================================
    # ============== CROSSOVER OPERATORS ==============
    # =================================================
    def randomTopTwoCrossover(self):
        """Crossover of the alpha chromosome at a random point"""
        pass

    # =================================================
    # ============== MUTATION OPERATORS ==============
    # =================================================
    def randomSinglePointMutation(self, individualNum: int):
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
    def visualizeIndividual(self, generation: str, individualRank: int):
        """Draws the .html file for the top individual"""
        self.rankPopulation()
        self.population[individualRank].visualizeAlphaNetwork(frontCatName=generation + "-")
