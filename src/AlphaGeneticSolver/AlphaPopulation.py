import random
import time

from src.AlphaGeneticSolver.AlphaIndividual import AlphaIndividual
from src.AlphaGeneticSolver.AlphaSolver import AlphaSolver
from src.AlphaGeneticSolver.AlphaVisualizer import AlphaVisualizer
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class AlphaPopulation:
    """Class that manages a population of alpha-reduced individuals and handles the genetic algorithm operators"""

    def __init__(self, FCFNinstance: FixedChargeFlowNetwork, targetFlow: int, populationSize: int, numGenerations: int):
        """Constructor of a Population instance"""
        # Input network and topology data (NOTE: Input network must be unsolved. If it's not, reload from disc.)
        if FCFNinstance.isSolved is False:
            self.FCFN = FCFNinstance
        else:
            unsolvedFCFN = FixedChargeFlowNetwork()
            unsolvedFCFN.loadFCFNfromDisc(FCFNinstance.name)
            self.FCFN = unsolvedFCFN
        # Population/GA attributes
        self.targetFlow = targetFlow
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.population = []

        # Initialize population
        for i in range(populationSize):
            thisIndividual = AlphaIndividual(self, i)
            thisIndividual.initializeAlphaValues(
                "random")  # Determines how alpha values are initialized at the beginning
            self.population.append(thisIndividual)

    def solvePopulation(self):
        """Solves the alpha-reduced LP of each individual in the population"""
        for individual in self.population:
            if individual.isSolved is False:
                solverLP = AlphaSolver(individual, self.targetFlow)
                solverLP.buildModel()
                solverLP.solveModel()
                solverLP.writeSolution()
                individual.calculateTrueCost()
                # visualAlpha = AlphaVisualize(individual)
                # visualAlpha.drawGraph(individual.name)

    def rankPopulation(self):
        """Ranks the population from least cost to greatest (i.e. fitness)"""
        self.population.sort(key=lambda x: x.trueCost, reverse=False)  # reverse=False ranks least to greatest
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
        offspringOne = AlphaIndividual(self, -1)
        offspringTwo = AlphaIndividual(self, -1)
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
        self.solvePopulation()
        self.rankPopulation()

    def randomSinglePointMutation(self, individual: AlphaIndividual):
        """Mutates an individual at a random gene in the chromosome"""
        random.seed()
        mutatePoint = random.randint(0, self.FCFN.numEdges - 1)
        individual.alphaValues[mutatePoint] = random.random()
        individual.solved = False

    def randomTotalMutation(self, individual: AlphaIndividual):
        """Mutates the entire chromosome of an individual"""
        individual.alphaValues = individual.initializeAlphaValuesRandomly()

    def visualizeTopTwo(self, generation: str):
        """Draws the .html file for the top two individuals"""
        visualBest = AlphaVisualizer(self.population[0])
        visualSecondBest = AlphaVisualizer(self.population[1])
        visualBest.drawGraph(self.population[0].name + "--" + generation + "--")
        visualSecondBest.drawGraph(self.population[1].name + "--" + generation + "--")

    def visualizeTop(self, generation: str):
        """Draws the .html file for the top individual"""
        visualBest = AlphaVisualizer(self.population[0])
        visualBest.drawGraph(self.population[0].name + "--" + generation + "--")

    def evolvePopulation(self):
        """Evolves the population based on the crossover and mutation operators"""
        self.solvePopulation()
        self.rankPopulation()
        for generation in range(0, self.numGenerations):
            # self.randomTopTwoCrossover()
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
