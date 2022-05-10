import sys

from src.AntColony.Ant import Ant
from src.Network.FlowNetwork import FlowNetwork
from src.Network.Solution import Solution
from src.Network.SolutionVisualizer import SolutionVisualizer


class Colony:
    """Class that defines a Colony object, representing an entire ant colony in the AntColony approach"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, network: FlowNetwork, minTargetFlow: float, numAnts: int, numEpisodes: int):
        """Constructor of a Colony instance"""
        # Input Attributes
        self.network = network
        self.minTargetFlow = minTargetFlow

        # Hyperparameters
        self.numEpisodes = numEpisodes  # One episode = All the ants completing one tour (i.e. creating a valid solution) each
        self.numAnts = numAnts  # Number of tours completed per episode
        self.initialPheromoneConcentration = 1000000  # Amount of pheromone initially deposited on arcs
        self.evaporationRate = 0.75  # rho = Rate at which pheromone is lost (NOTE: 1 = complete loss/episode; 0 = no loss/episode)
        self.alpha = 1  # alpha = Relative importance to the ant of pheromone over "goodness" of arc
        self.beta = 1  # beta = Relative importance to the ant of "goodness" of arc over pheromone
        self.Q = 20  # Q = Proportionality scalar of best solution, which scales how much pheromone the best solution deposits

        # Colony Attributes
        self.population = self.initializePopulation()  # Contains the population of ants
        self.pheromoneDict = self.initializePheromoneDict()  # Dictionary indexed on key (fromNode, toNode, cap) with value (pheromone deposited)
        self.goodnessDict = self.initializeGoodnessOfArcDict()  # Dictionary indexed on key (fromNode, toNode, cap) with value (eta) (i.e. the "goodness" of taking that arc)
        self.bestKnownCost = None  # Stores the lowest cost solution found so far
        self.bestKnownSolution = None  # Stores the global best solution found so far
        self.convergenceData = []  # Stores the best known cost after each episode of the colony
        self.visual = None  # Object used to view the best solutions of the episode over time

    def solveNetwork(self, drawing=True, labels=True) -> Solution:
        """Main loop that solves the Flow Network instance with the AntColony"""
        # EPISODE LOOP
        for episode in range(self.numEpisodes):
            # print("\nStarting Episode " + str(episode) + "...")  # PRINT OPTION
            # INDIVIDUAL ANT EXPLORATION LOOP
            for antIndex in range(self.numAnts):
                # print("Solving ant " + str(antIndex) + "...")  # PRINT OPTION
                # In series, solve each ant one at a time
                self.population[antIndex].findSolution(self.pheromoneDict, self.goodnessDict)
            # POST-EXPLORATION DAEMON UPDATES
            # print("Doing post-exploration updates...")  # PRINT OPTION
            self.updateBestSolution()  # Updates the best solution only if this population contains it
            self.convergenceData.append(self.bestKnownCost)  # Save the best known cost at this episode
            self.evaporatePheromone()  # Reduces the pheromone across the entire dictionary based on rho
            self.depositPheromone()  # Deposits new pheromone on the arcs in the best known solution
            self.resetAllAnts()  # Clears the tour/solution attributes of every ant in the population for the next episode
            if drawing is True:
                self.visual = SolutionVisualizer(self.bestKnownSolution)  # Instantiate a visualizer
                if labels is True:
                    self.visual.drawGraphWithLabels(leadingText="Ep" + str(episode) + "_")  # Draw graph w/ labels
                else:
                    self.visual.drawUnlabeledGraph(leadingText="Ep" + str(episode) + "_")  # Draw graph w/o labels
        return self.bestKnownSolution  # Should return the best solution found at the end

    def updateBestSolution(self) -> None:
        """Finds the best solution in the current population and updates the global best if necessary"""
        currentBestCost = sys.maxsize
        currentBestAnt = None
        # Iterate over the current population to find current best
        for ant in self.population:
            if ant.trueCost < currentBestCost:
                currentBestCost = ant.trueCost
                currentBestAnt = ant
        # Compare current best to global best
        if self.bestKnownCost is None:
            self.bestKnownCost = currentBestCost
            self.bestKnownSolution = currentBestAnt.writeSolution()
        elif currentBestCost < self.bestKnownCost:
            self.bestKnownCost = currentBestCost
            self.bestKnownSolution = currentBestAnt.writeSolution()

    def evaporatePheromone(self) -> None:
        """Evaporates pheromone using (1-rho)*pheromone across the entire dictionary"""
        for arc in self.pheromoneDict.keys():
            self.pheromoneDict[arc] = self.pheromoneDict[arc] * (1 - self.evaporationRate)

    def depositPheromone(self) -> None:
        """Deposits new pheromone on the arcs contained in the best known solution so far"""
        # Deposit pheromone on edges
        for edgeIndex in range(self.network.numEdges):
            for capIndex in range(self.network.numArcCaps):
                edge = self.network.edgesArray[edgeIndex]
                cap = self.network.possibleArcCapsArray[capIndex]
                arcFlow = self.bestKnownSolution.arcFlows[(edgeIndex, capIndex)]
                if arcFlow > 0.0:
                    # OLD: self.pheromoneDict[(edge[0], edge[1], cap)] += self.Q / self.bestKnownCost
                    self.pheromoneDict[(edge[0], edge[1], cap)] += (self.Q * arcFlow) / self.bestKnownCost
        # Deposit pheromone on sources
        for sourceIndex in range(self.network.numSources):
            source = self.network.sourcesArray[sourceIndex]
            srcFlow = self.bestKnownSolution.sourceFlows[sourceIndex]
            if srcFlow > 0.0:
                srcKey = (-1, source, self.network.sourceCapsArray[sourceIndex])
                # OLD: self.pheromoneDict[srcKey] += self.Q / self.bestKnownCost
                self.pheromoneDict[srcKey] += (self.Q * srcFlow) / self.bestKnownCost
        # Deposit pheromone on sinks
        for sinkIndex in range(self.network.numSinks):
            sink = self.network.sinksArray[sinkIndex]
            sinkFlow = self.bestKnownSolution.sinkFlows[sinkIndex]
            if sinkFlow > 0.0:
                sinkKey = (sink, -2, self.network.sinkCapsArray[sinkIndex])
                # OLD: self.pheromoneDict[sinkKey] += self.Q / self.bestKnownCost
                self.pheromoneDict[sinkKey] += (self.Q * sinkFlow) / self.bestKnownCost

    def initializePopulation(self) -> list:
        """Initializes the population with ants objects"""
        population = []
        for n in range(self.numAnts):
            thisAnt = Ant(self.network, self.minTargetFlow, self.alpha, self.beta)
            population.append(thisAnt)
        return population

    def initializePheromoneDict(self) -> dict:
        """Adds all possible arcs and supersource/sink as keys to the pheromone dictionary"""
        pheromoneDict = {}
        # For all edge, cap pairs, initialize with one
        for edge in self.network.edgesArray:
            for cap in self.network.possibleArcCapsArray:
                pheromoneDict[(edge[0], edge[1], cap)] = self.initialPheromoneConcentration
        # For all supersource -> source and visa versa, initialize with zero
        for srcIndex in range(self.network.numSources):
            source = self.network.sourcesArray[srcIndex]
            cap = self.network.sourceCapsArray[srcIndex]
            pheromoneDict[(-1, source, cap)] = self.initialPheromoneConcentration
            pheromoneDict[(source, -1, -1)] = 0.0
        # For all supersink -> sink, initialize with zero (NOTE: You can't go back from a supersink)
        for sinkIndex in range(self.network.numSinks):
            sink = self.network.sinksArray[sinkIndex]
            cap = self.network.sinkCapsArray[sinkIndex]
            pheromoneDict[(sink, -2, cap)] = self.initialPheromoneConcentration
        return pheromoneDict

    def initializeGoodnessOfArcDict(self) -> dict:
        """Adds all possible arcs and supersource/sink as keys to the pheromone dictionary with a value of cap/(FixedCost + VariableCost*cap)"""
        arcGoodnessScalar = 10  # Based off the magnitude of the arc costs (~10^2)
        arcGoodnessDict = {}
        # For all edge, cap pairs, initialize with one
        for edge in self.network.edgesArray:
            for cap in self.network.possibleArcCapsArray:
                arcObj = self.network.arcsDict[(edge[0], edge[1], cap)]
                # OLD: arcGoodness = arcGoodnessScalar / (arcObj.fixedCost + arcObj.variableCost)
                # OLD: arcGoodness = (arcGoodnessScalar * cap) / (arcObj.fixedCost + arcObj.variableCost)
                arcGoodness = (arcGoodnessScalar * cap) / (arcObj.fixedCost + arcObj.variableCost * cap)
                arcGoodnessDict[(edge[0], edge[1], cap)] = arcGoodness
        # For all supersource -> source and visa versa, initialize with zero
        for srcIndex in range(self.network.numSources):
            source = self.network.sourcesArray[srcIndex]
            cap = self.network.sourceCapsArray[srcIndex]
            variableCost = self.network.sourceVariableCostsArray[srcIndex]
            # OLD: srcGoodness = arcGoodnessScalar / variableCost
            # OLD: srcGoodness = (cap * arcGoodnessScalar) / variableCost
            srcGoodness = (cap * arcGoodnessScalar) / (variableCost * cap)
            arcGoodnessDict[(-1, source, cap)] = srcGoodness
            arcGoodnessDict[(source, -1, -1)] = 0.0  # NOTE: MAKES THE "GOODNESS" OF MOVING SOURCE -> SUPERSOURCE ZERO
        # For all supersink -> sink, initialize with zero (NOTE: You can't go back from a supersink)
        for sinkIndex in range(self.network.numSinks):
            sink = self.network.sinksArray[sinkIndex]
            cap = self.network.sinkCapsArray[sinkIndex]
            variableCost = self.network.sinkVariableCostsArray[sinkIndex]
            # OLD: sinkGoodness = arcGoodnessScalar * 10 / variableCost
            # OLD: sinkGoodness = (cap * arcGoodnessScalar ** 2) / variableCost
            sinkGoodness = (cap * arcGoodnessScalar ** 2) / (
                        variableCost * cap)  # SCALAR^2 CREDITS SINK -> SUPERSINK MOVES
            arcGoodnessDict[(sink, -2, cap)] = sinkGoodness
        return arcGoodnessDict

    def resetAllAnts(self) -> None:
        """Resets the tour/solution attributes for all the ants in the population"""
        for ant in self.population:
            ant.resetTourAndSolutionAttributes()
