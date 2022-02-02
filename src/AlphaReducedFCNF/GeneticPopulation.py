import random

from src.AlphaReducedFCNF.AlphaFCNF import AlphaFCNF
from src.AlphaReducedFCNF.AlphaLP import AlphaLP
from src.FixedCostNetworkFlowSolver.FCNF import FCNF


class GeneticPopulation:
    """Class that manages a population of alpha-reduced genetic algorithms"""

    def __init__(self, FCNFinstance: FCNF, targetFlow: int, populationSize: int, numGenerations: int):
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
        for individual in self.population:
            print(individual.totalCost)

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
        # TODO - Resolve and resort???

    def randomMutation(self, individual: AlphaFCNF):
        """Mutates an individual at a random gene in the chromosome"""
        pass

    def evolvePopulation(self):
        """Evolves the population based on the crossover and mutation operators"""
        pass
