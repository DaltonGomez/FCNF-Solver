import random

import numpy as np
from numpy import ndarray

from src.AlphaGenetic.AlphaSolverPDLP import AlphaSolverPDLP
from src.AlphaGenetic.Individual import Individual
from src.Network.FlowNetwork import FlowNetwork
from src.Network.Solution import Solution
from src.Network.SolutionVisualizer import SolutionVisualizer


class Population:
    """Class that manages a population of alpha-relaxed individuals and handles their evolution with GA operators"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, network: FlowNetwork, minTargetFlow: float, populationSize: int):
        """Constructor of a Population instance"""
        # Input Attributes
        self.network = network
        self.minTargetFlow = minTargetFlow
        # Population Attributes & Solver
        self.populationSize = populationSize
        self.numGenerations = 1
        self.population = []
        self.initializePopulation()  # Initialize population
        self.solver = AlphaSolverPDLP(self.network, self.minTargetFlow)  # Pre-build variables/constraints in solver

    # ============================================
    # ============== EVOLUTION LOOP ==============
    # ============================================
    def evolvePopulation(self, numGenerations, drawing=False) -> None:
        """Evolves the population for a specified number of generations"""
        self.numGenerations = numGenerations
        for generation in range(numGenerations):
            # TODO - SELECTION & CROSSOVER
            # TODO - SELECTION & MUTATION
            # Solve and visualize (NAIVE HILL CLIMB CURRENTLY AS POC)
            self.naiveHillClimb()
            self.solvePopulation()
            if drawing is True:
                self.visualizeBestIndividual(labels=False, leadingText=str(generation))
            print("Generation = " + str(generation) + "\tBest Individual = " + str(self.population[0].trueCost))

    # ==============================================
    # ============== MUTATION METHODS ==============
    # ==============================================
    def naiveHillClimb(self) -> None:
        """Sorts the population by rank and hypermutates the worst individual only at each generation"""
        self.population = self.rankPopulation()
        for i in range(1, self.populationSize):
            self.hypermutateIndividual(self.population[i])

    def hypermutatePopulation(self) -> None:
        """Reinitializes the entire population (i.e. an extinction event with a brand new population spawned)"""
        for individual in self.population:
            self.hypermutateIndividual(individual)

    def hypermutateIndividual(self, individual: Individual) -> None:
        """Reinitializes the individual's entire alpha values (i.e. kills them off and spawns a new individual)"""
        newAlphas = self.getInitialAlphaValues()
        individual.setAlphaValues(newAlphas)
        individual.resetOutputNetwork()

    # ====================================================
    # ============== INITIALIZATION METHODS ==============
    # ====================================================
    def initializePopulation(self) -> None:
        """Initializes the GA population with random alpha values"""
        for individual in range(self.populationSize):
            thisGenotype = self.getInitialAlphaValues()
            thisIndividual = Individual(thisGenotype)
            self.population.append(thisIndividual)

    def getInitialAlphaValues(self) -> ndarray:
        """Returns a randomly initialized array of alpha values (i.e. the genotype)"""
        tempAlphaValues = []
        for edge in range(self.network.numEdges):
            tempEdge = []
            for cap in range(self.network.numArcCaps):
                thisAlphaValue = self.getRandomAlphaValue()
                tempEdge.append(thisAlphaValue)
            tempAlphaValues.append(tempEdge)
        initialGenotype = np.array(tempAlphaValues)
        return initialGenotype

    @staticmethod
    def getRandomAlphaValue() -> float:
        """Returns a single alpha value for population initialization"""
        # TODO - Tune this method (i.e. the probability distribution/parameters best for population initialization)
        random.seed()
        randomGene = random.random()
        return randomGene

    # ============================================
    # ============== HELPER METHODS ==============
    # ============================================
    def rankPopulation(self) -> list:
        """Ranks the population in ascending order of true cost (i.e. Lower cost -> More fit) and returns"""
        sortedPopulation = sorted(self.population, key=lambda x: x.trueCost)
        return sortedPopulation

    def getMostFitIndividual(self) -> Individual:
        """Returns the most fit individual in the population"""
        sortedPop = self.rankPopulation()
        return sortedPop[0]

    # ============================================
    # ============== SOLVER METHODS ==============
    # ============================================
    def solvePopulation(self) -> None:
        """Solves all unsolved instances in the entire population"""
        for individual in self.population:
            if individual.isSolved is False:
                self.solveIndividual(individual)

    def solveIndividual(self, individual: Individual) -> None:
        """Solves a single individual and writes the expressed network to the individual"""
        # Overwrite new objective function with new alpha values and solve
        self.solver.updateObjectiveFunction(individual.alphaValues)
        self.solver.solveModel()
        # Write expressed network output data to individual
        individual.isSolved = True
        individual.arcFlows = self.solver.getArcFlowsDict()
        individual.arcsOpened = self.solver.getArcsOpenDict()
        individual.srcFlows = self.solver.getSrcFlowsList()
        individual.sinkFlows = self.solver.getSinkFlowsList()
        individual.trueCost = self.solver.calculateTrueCost()
        individual.fakeCost = self.solver.getObjectiveValue()
        # Reset solver
        self.solver.resetSolver()

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def visualizeBestIndividual(self, labels=False, leadingText="") -> None:
        """Renders the visualization for the most fit individual in the population at any time"""
        bestIndividual = self.getMostFitIndividual()
        self.visualizeIndividual(bestIndividual, labels=labels, leadingText=leadingText)

    def visualizeAllIndividuals(self, labels=False, leadingText="") -> None:
        """Renders the visualization for all individuals in the population at any time"""
        i = 0
        for individual in self.population:
            self.visualizeIndividual(individual, labels=labels, leadingText=leadingText + "_" + str(i))
            i += 1

    def visualizeIndividual(self, individual: Individual, labels=False, leadingText="") -> None:
        """Renders the visualization for a specified individual"""
        solution = Solution(self.network, self.minTargetFlow, individual.fakeCost, individual.trueCost,
                            individual.srcFlows, individual.sinkFlows, individual.arcFlows, individual.arcsOpened,
                            "alphaGA", False, self.network.isSourceSinkCapacitated, self.network.isSourceSinkCharged)
        visualizer = SolutionVisualizer(solution)
        if labels is True:
            visualizer.drawGraphWithLabels(leadingText=leadingText)
        else:
            visualizer.drawUnlabeledGraph(leadingText=leadingText)
