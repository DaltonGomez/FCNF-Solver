import sys

from src.AlphaReducedFCNF.AlphaFCNF import AlphaFCNF
from src.AlphaReducedFCNF.AlphaLP import AlphaLP
from src.AlphaReducedFCNF.AlphaVisualize import AlphaVisualize
from src.FixedCostNetworkFlowSolver.FCNF import FCNF


class GeneticPopulation:
    """Class that manages a population of alpha-reduced genetic algorithms"""

    def __init__(self, FCNFinstance: FCNF, targetFlow: int, populationSize: int, numGenerations: int):
        """Constructor of a GeneticPopulation instance"""
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
            solverLP = AlphaLP(individual, self.targetFlow)
            solverLP.buildModel()
            solverLP.solveModel()
            solverLP.writeSolution()
            individual.calculateTrueCost()

    def getBestIndividual(self):
        """Returns the best individual from the population"""
        bestIndividual = None
        smallestCost = int(sys.maxsize)
        for individual in self.population:
            if individual.totalCost < smallestCost:
                bestIndividual = individual
        visualAlpha = AlphaVisualize(bestIndividual)
        visualAlpha.drawGraph(bestIndividual.name)

    def evolvePopulation(self):
        """Evolves the population based on the crossover and mutation operators"""
        pass

    def randomCrossover(self):
        """Crossover of the alpha-reduced chromosome at a random point"""
        pass

    def randomMutation(self):
        """Mutates an individual at a random gene in the chromosome"""
        pass
