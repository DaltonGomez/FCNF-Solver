import random
import time

from src.AlphaReducedFCNF.AlphaFCNF import AlphaFCNF
from src.AlphaReducedFCNF.AlphaLP import AlphaLP
from src.AlphaReducedFCNF.AlphaVisualize import AlphaVisualize
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class GeneticPopulation:
    """Class that manages a population of alpha-reduced genetic algorithms"""

    def __init__(self, FCNFinstance: FixedChargeFlowNetwork, targetFlow: int, populationSize: int, numGenerations: int):
        """Constructor of a GeneticPopulation instance"""
        self.FCNF = FCNFinstance
        self.targetFlow = targetFlow
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.population = []
        for i in range(populationSize):
            thisIndividual = AlphaFCNF(FCNFinstance)
            thisIndividual.initializeAlphaValuesRandomly()
            self.population.append(thisIndividual)

    def solvePopulation(self):
        """Solves the alpha-reduce LP of each individual in the population"""
        for individual in self.population:
            if individual.solved is False:
                solverLP = AlphaLP(individual, self.targetFlow)
                solverLP.buildModel()
                solverLP.solveModel()
                solverLP.writeSolution()
                individual.calculateTrueCost()
                # visualAlpha = AlphaVisualize(individual)
                # visualAlpha.drawGraph(individual.name)

    def rankPopulation(self):
        """Ranks the population from least cost to greatest (i.e. fitness)"""
        self.population.sort(key=lambda x: x.totalCost, reverse=False)  # reverse=False ranks least to greatest
        # for individual in self.population:
        #    print(individual.totalCost)

    def randomTopTwoCrossover(self):
        """Crossover of the alpha-reduced chromosome at a random point"""
        # Rank population and discard bottom two individuals
        self.rankPopulation()
        self.population.pop(-1)
        self.population.pop(-1)
        # Get crossover point and direction from RNG
        random.seed()
        crossoverPoint = random.randint(0, self.population[0].FCNF.numEdges)
        crossoverDirection = random.randint(0, 2)
        # Initialize offspring
        offspringOne = AlphaFCNF(self.FCNF)
        offspringTwo = AlphaFCNF(self.FCNF)
        # Conduct crossover
        if crossoverDirection == 0:
            for i in range(crossoverPoint):
                offspringOne.alphaValues.append(self.population[0].alphaValues[i])
                offspringTwo.alphaValues.append(self.population[1].alphaValues[i])
            for j in range(crossoverPoint, self.FCNF.numEdges):
                offspringOne.alphaValues.append(self.population[0].alphaValues[j])
                offspringTwo.alphaValues.append(self.population[1].alphaValues[j])
        else:
            for i in range(crossoverPoint):
                offspringOne.alphaValues.append(self.population[1].alphaValues[i])
                offspringTwo.alphaValues.append(self.population[0].alphaValues[i])
            for j in range(crossoverPoint, self.FCNF.numEdges):
                offspringOne.alphaValues.append(self.population[1].alphaValues[j])
                offspringTwo.alphaValues.append(self.population[0].alphaValues[j])
        # Add offspring into population
        self.population.append(offspringOne)
        self.population.append(offspringTwo)
        self.solvePopulation()
        self.rankPopulation()

    def randomSinglePointMutation(self, individual: AlphaFCNF):
        """Mutates an individual at a random gene in the chromosome"""
        random.seed()
        mutatePoint = random.randint(0, self.FCNF.numEdges - 1)
        individual.alphaValues[mutatePoint] = random.random()
        individual.solved = False

    def randomTotalMutation(self, individual: AlphaFCNF):
        """Mutates the entire chromosome of an individual"""
        individual.alphaValues = individual.initializeAlphaValuesRandomly()

    def visualizeTopTwo(self, generation: str):
        """Draws the .html file for the top two individuals"""
        visualBest = AlphaVisualize(self.population[0])
        visualSecondBest = AlphaVisualize(self.population[1])
        visualBest.drawGraph(self.population[0].name + "--" + generation + "--")
        visualSecondBest.drawGraph(self.population[1].name + "--" + generation + "--")

    def visualizeTop(self, generation: str):
        """Draws the .html file for the top individual"""
        visualBest = AlphaVisualize(self.population[0])
        visualBest.drawGraph(self.population[0].name + "--" + generation + "--")

    def evolvePopulation(self):
        """Evolves the population based on the crossover and mutation operators"""
        self.solvePopulation()
        self.rankPopulation()
        for generation in range(0, self.numGenerations):
            self.randomTopTwoCrossover()
            random.seed()
            for individual in self.population:
                # Do not mutate the top two individuals
                if individual == self.population[0] or individual == self.population[1]:
                    continue
                # TODO - Modify so that mutation rates are not hardcoded
                elif random.random() < 0.25:
                    self.randomSinglePointMutation(individual)
                elif random.random() < 0.05:
                    self.randomTotalMutation(individual)
            self.visualizeTop(str(generation))
            # Timeout to ensure correct visualization rendering order
            time.sleep(0.2)
        return self.population[0]
