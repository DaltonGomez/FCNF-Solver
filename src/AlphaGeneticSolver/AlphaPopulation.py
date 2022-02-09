import random

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
            unsolvedFCFN.loadFCFNfromDisc(FCFNinstance.name)
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
    def evolvePopulation(self):
        """Evolves the population based on the crossover and mutation operators"""
        random.seed()
        for generation in range(self.numGenerations):
            self.solvePopulation()
            self.rankPopulation()
            print("Generation= " + str(generation) + "\tBest Individual= " + str(self.population[0].trueCost))
            self.randomTopTwoCrossover()
            for individual in range(self.populationSize):
                if individual == self.population[0] or individual == self.population[1]:
                    continue  # Don't mutate the top two individuals
                # TODO - Modify so that mutation rates are not hardcoded
                elif random.random() < 0.25:
                    self.randomSinglePointMutation(individual)
                elif random.random() < 0.05:
                    self.randomTotalMutation(individual)
            # Visualization w/ timeout to ensure correct visualization rendering order
            # self.visualizeTop(str(generation))
            # time.sleep(0.5)

    # =================================================
    # ============== CROSSOVER OPERATORS ==============
    # =================================================
    def randomTopTwoCrossover(self):
        """Crossover of the alpha chromosome at a random point"""
        # Rank population and discard bottom two individuals
        self.rankPopulation()
        self.population.pop(-1)
        self.population.pop(-1)
        # Get crossover point and direction from RNG
        random.seed()
        crossoverPoint = random.randint(0, self.FCFN.numEdges)
        crossoverDirection = random.randint(0, 2)
        # Initialize offspring
        offspringOne = AlphaIndividual(self.FCFN)
        offspringTwo = AlphaIndividual(self.FCFN)
        # Conduct crossover
        if crossoverDirection == 0:
            for i in range(crossoverPoint):
                offspringOne.alphaValues.append(self.population[0].alphaValues[i])
                offspringTwo.alphaValues.append(self.population[1].alphaValues[i])
            for j in range(crossoverPoint, self.FCFN.numEdges):
                offspringOne.alphaValues.append(self.population[0].alphaValues[j])
                offspringTwo.alphaValues.append(self.population[1].alphaValues[j])
        else:
            for i in range(crossoverPoint):
                offspringOne.alphaValues.append(self.population[1].alphaValues[i])
                offspringTwo.alphaValues.append(self.population[0].alphaValues[i])
            for j in range(crossoverPoint, self.FCFN.numEdges):
                offspringOne.alphaValues.append(self.population[1].alphaValues[j])
                offspringTwo.alphaValues.append(self.population[0].alphaValues[j])
        # Add offspring into population
        self.population.append(offspringOne)
        self.population.append(offspringTwo)

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
    def visualizeTop(self, generation: str):
        """Draws the .html file for the top individual"""
        self.rankPopulation()
        self.population[0].visualizeAlphaNetwork(frontCatName=generation + "-")
