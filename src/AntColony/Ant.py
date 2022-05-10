import random
import sys

from src.Network.FlowNetwork import FlowNetwork
from src.Network.Solution import Solution


class Ant:
    """Class that defines an AntColony object, representing a single agent exploring the flow network space"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, network: FlowNetwork, minTargetFlow: float, alpha: float, beta: float):
        """Constructor of an AntColony instance"""
        # Input Attributes
        self.network = network  # Input network data
        self.minTargetFlow = minTargetFlow  # Target flow to assign
        self.pheromoneDict = {}  # Dictionary indexed on key (fromNode, toNode, cap) with value (pheromone deposited in previous episodes)
        self.goodnessDict = {}  # Dictionary indexed on key (fromNode, toNode, cap) with value (eta) (i.e. the "goodness" of taking that arc)

        # Hyperparameters
        self.alpha = alpha  # alpha = Relative importance to the ant of pheromone over "goodness" of arc
        self.beta = beta  # beta = Relative importance to the ant of "goodness" of arc over pheromone

        # Tour Attributes
        # NOTE: A "tour" is a complete feasible solution that assigns all target flow across the network in a number of trips
        self.time = 0  # Incremented every time an arc is traveled (i.e. measure of the total search time for the ant to complete a tour)
        self.numTrips = 0  # Number of trips the ant has taken
        self.currentPosition = -1  # Node ID of the ant's position- NOTE: -1 Represents the supersource and -2 represents the supersink
        self.remainingFlowToAssign = self.minTargetFlow  # "Mountain" of flow initially at the supersource that the ant has to move
        self.assignedFlowDict = self.initializeAssignedFlowDict()  # Dictionary indexed on key (fromNode, toNode, cap) with value (cumulative flow assigned)
        self.availableCapacityDict = self.initializeAvailableCapacityDict()  # Dictionary indexed on key (fromNode, toNode, cap) with value (available capacity until full)

        # Trip Attributes
        # NOTE: A "trip" is a single supersource -> supersink path that assigns only x amount of flow
        self.tripStack = []  # Maintains the arcs traveled on the current trip (NOTE: Should be treated as a true stack- push/pop only!)
        self.nodesVisitedThisTrip = set()  # AntColony's memory of visited nodes this trip (Used for cycle/backtracking detection)

        # Solution Attributes (Written after an ant completes a tour)
        self.trueCost = 0.0
        self.sourceFlows = []
        self.sinkFlows = []
        self.arcFlows = {}
        self.arcsOpened = {}

    def findSolution(self, pheromoneDict: dict, goodnessDict: dict) -> None:
        """Main loop that has the ant explore the graph space until a feasible solution is discovered"""
        # Update dictionaries for determining edge selection before solving
        self.pheromoneDict = pheromoneDict  # Dictionary indexed on key (fromNode, toNode, cap) with value (pheromone deposited in previous episodes)
        self.goodnessDict = goodnessDict  # Dictionary indexed on key (fromNode, toNode, cap) with value (eta) (i.e. the "goodness" of taking that arc)
        # TOUR LOOP
        while self.remainingFlowToAssign > 0.0:  # While all flow is not delivered
            # PRE-TRIP SETUP
            self.resetTripAttributes()  # Start a new trip
            # TRIP LOOP
            while self.currentPosition != -2:  # Explore until the supersink is found
                options = self.getPossibleNextMoves()  # Get options for next move by checking non-full adjacent edges
                arcChoice = self.decideArcToTraverse(options)  # Probabilistically choose a next arc
                # self.printTimeStepData(arcChoice)  # PRINT OPTION
                self.travelArc(arcChoice)  # Move the ant across the arc and assign flow
            # POST-TRIP ACCOUNTING
            self.assignTripFlow()  # Assigns flow to all arcs traveled in the trip, where the amount is equal to the minimum available capacity seen
            self.numTrips += 1  # Increment trips
            # self.printTripData()  # PRINT OPTION
            if self.time > 100000:  # Restart if timed out
                print("Restarting ant!")
                self.resetTourAndSolutionAttributes()
        self.resolveOpposingFlows()  # Eliminates positive flows in opposing directions on every bidirectional edge
        self.computeResultingNetwork()  # Calculates the cost and data structures for writing to a solution object
        # print("Solution Cost = " + str(self.trueCost) + "\n")  # PRINT OPTION

    def getPossibleNextMoves(self) -> list:
        """Returns the possible options the ant could take on their next timestep"""
        options = []
        # Evaluate options if you're at supersource
        if self.currentPosition == -1:
            for srcIndex in range(self.network.numSources):
                source = self.network.sourcesArray[srcIndex]
                cap = self.network.sourceCapsArray[srcIndex]
                arcID = (-1, source, cap)
                if self.availableCapacityDict[arcID] > 0.0:
                    options.append((-1, source, cap))
        # Evaluate options if you're anywhere else
        else:
            nodeObj = self.network.nodesDict[self.currentPosition]
            # Add all arcs that are not at capacity to the options
            for edge in nodeObj.outgoingEdges:
                for cap in self.network.possibleArcCapsArray:
                    arcID = (edge[0], edge[1], cap)
                    # Previous is was: if self.availableCapacity[arcID] > 0.0 and arcID not in self.arcsVisitedThisTrip:
                    if self.availableCapacityDict[arcID] > 0.0:
                        options.append((edge[0], edge[1], cap))
            # Give the possibility to go back to the supersource if at a source
            if nodeObj.nodeType == 0:
                options.append((nodeObj.nodeID, -1, -1))
            # Give the possibility to go to the supersink if at a sink
            elif nodeObj.nodeType == 1:
                for sinkIndex in range(self.network.numSinks):
                    if nodeObj.nodeID == self.network.sinksArray[sinkIndex]:
                        options.append((nodeObj.nodeID, -2, self.network.sinkCapsArray[sinkIndex]))
        return options

    def decideArcToTraverse(self, options: list) -> tuple:
        """Probabilistically selects the arc the ant will travel on the next timestep"""
        random.seed()
        # Check if the only option is to move back to the supersource, and if so, return that option
        if len(options) == 1 and options[0][1] == -1:
            return options[0]
        # Compute numerators and denominators
        numerators = []
        denominator = 0.0
        for arc in options:
            thisArcsNumerator = (self.pheromoneDict[arc] ** self.alpha) * (self.goodnessDict[arc] ** self.beta)
            numerators.append(thisArcsNumerator)
            denominator += thisArcsNumerator
        # Build cumulative probability distribution
        cumulativeProbabilities = [numerators[0] / denominator]
        for i in range(1, len(numerators)):
            cumulativeProbabilities.append((numerators[i] / denominator) + cumulativeProbabilities[i - 1])
        # Roll RNG and select edge
        rng = random.random()
        # self.printProbabilityDistribution(rng, options, cumulativeProbabilities)  # PRINT OPTION
        for arc in range(len(options)):
            if rng < cumulativeProbabilities[arc]:
                return options[arc]

    def travelArc(self, arcChoice: tuple) -> None:
        """Moves the ant across an arc, using the ant's memory to detect and undo cycles and backtracks"""
        self.time += 1  # Increment the timestep
        # If the move is a back track, then undo the flow assigned on the last step
        if self.isVisitedNode(arcChoice) is True:
            while self.tripStack[-1][1] != arcChoice[1]:
                poppedPrevMove = self.tripStack.pop(-1)  # Pop previous move off trip stack
                self.currentPosition = poppedPrevMove[0]  # Update position by undoing popped move
                self.nodesVisitedThisTrip.remove(poppedPrevMove[1])  # Remove the popped nodes from the ant's memory
        else:
            self.currentPosition = arcChoice[1]  # Update position (i.e. move across arc)
            self.nodesVisitedThisTrip.add(arcChoice[1])  # Adds the node to the ant's memory of nodes this trip
            self.tripStack.append(arcChoice)  # Push the move onto the trip stack

    def isVisitedNode(self, arcChoice: tuple) -> bool:
        """Checks the set of already visited nodes to identify cycles/backtracks"""
        if arcChoice[1] in self.nodesVisitedThisTrip:
            return True
        else:
            return False

    def assignTripFlow(self) -> None:
        """Assigns flow to all arcs traveled, where the amount is the minimum available capacity seen on the trip"""
        # Get bottleneck size (i.e. minimum available capacity over all arcs visited on trip)
        minAvailableCapacityDuringTrip = self.remainingFlowToAssign  # Start the flow to assign at the remaining amount
        for arc in self.tripStack:
            if self.availableCapacityDict[arc] < minAvailableCapacityDuringTrip:
                minAvailableCapacityDuringTrip = self.availableCapacityDict[arc]
        # Iterate over arcs, assigning flow and deducting from the minimum available capacity
        for arc in self.tripStack:
            self.availableCapacityDict[arc] -= minAvailableCapacityDuringTrip
            self.assignedFlowDict[arc] += minAvailableCapacityDuringTrip
        # Deduct assigned flow from the flow left to assign
        self.remainingFlowToAssign -= minAvailableCapacityDuringTrip

    def resolveOpposingFlows(self) -> None:
        """Iterates over resulting network and reduces opposing flows to be unidirectional"""
        # For each edge
        for edgeIndex in range(self.network.numEdges):
            forwardFlow = 0
            backwardFlow = 0
            edge = self.network.edgesArray[edgeIndex]
            # Find the total forward and backward flow
            for capIndex in range(self.network.numArcCaps):
                cap = self.network.possibleArcCapsArray[capIndex]
                forwardFlow += self.assignedFlowDict[(edge[0], edge[1], cap)]
                backwardFlow += self.assignedFlowDict[(edge[1], edge[0], cap)]
            # Resolve if both non-zero and equal to one another
            if 0 < forwardFlow == backwardFlow > 0:
                for capIndex in range(self.network.numArcCaps):
                    cap = self.network.possibleArcCapsArray[capIndex]
                    self.assignedFlowDict[(edge[0], edge[1], cap)] = 0
                    self.assignedFlowDict[(edge[1], edge[0], cap)] = 0
            # Resolve if both non-zero and forward flow is greater than backward flow
            # TODO - This (and the complementary condition below) will probably error out on flow distributed across parallel arcs
            elif forwardFlow > 0 and 0 < backwardFlow < forwardFlow:
                netFlow = forwardFlow - backwardFlow
                for capIndex in range(self.network.numArcCaps):
                    cap = self.network.possibleArcCapsArray[capIndex]
                    self.assignedFlowDict[(edge[0], edge[1], cap)] = netFlow
                    self.assignedFlowDict[(edge[1], edge[0], cap)] = 0
            # Resolve if both non-zero and forward flow is less than backward flow
            elif 0 < forwardFlow < backwardFlow and backwardFlow > 0:
                netFlow = backwardFlow - forwardFlow
                for capIndex in range(self.network.numArcCaps):
                    cap = self.network.possibleArcCapsArray[capIndex]
                    self.assignedFlowDict[(edge[0], edge[1], cap)] = 0
                    self.assignedFlowDict[(edge[1], edge[0], cap)] = netFlow

    def computeResultingNetwork(self) -> None:
        """Calculates the cost of the ant's solution"""
        trueCost = 0.0
        # Calculate source costs
        for sourceIndex in range(self.network.numSources):
            sourceID = self.network.sourcesArray[sourceIndex]
            sourceCapacity = self.network.sourceCapsArray[sourceIndex]
            sourceFlow = self.assignedFlowDict[(-1, sourceID, sourceCapacity)]
            self.sourceFlows.append(int(sourceFlow))
            sourceVariableCost = self.network.sourceVariableCostsArray[sourceIndex]
            trueCost += sourceVariableCost * sourceFlow
        # Calculate sink costs
        for sinkIndex in range(self.network.numSinks):
            sinkID = self.network.sinksArray[sinkIndex]
            sinkCapacity = self.network.sinkCapsArray[sinkIndex]
            sinkFlow = self.assignedFlowDict[(sinkID, -2, sinkCapacity)]
            self.sinkFlows.append(int(sinkFlow))
            sinkVariableCost = self.network.sinkVariableCostsArray[sinkIndex]
            trueCost += sinkVariableCost * sinkFlow
        # Calculate edge costs
        for edgeIndex in range(self.network.numEdges):
            for capIndex in range(self.network.numArcCaps):
                edge = self.network.edgesArray[edgeIndex]
                cap = self.network.possibleArcCapsArray[capIndex]
                arcObj = self.network.arcsDict[(edge[0], edge[1], cap)]
                arcFlow = self.assignedFlowDict[(edge[0], edge[1], cap)]
                self.arcFlows[(edgeIndex, capIndex)] = int(arcFlow)
                self.arcsOpened[(edgeIndex, capIndex)] = 0
                if arcFlow > 0:
                    trueCost += arcObj.fixedCost + arcObj.variableCost * arcFlow
                    self.arcsOpened[(edgeIndex, capIndex)] = 1
        self.trueCost = trueCost

    def writeSolution(self) -> Solution:
        """Writes the single ant's solution to a Solution instance for visualization/saving"""
        solution = Solution(self.network, self.minTargetFlow, self.trueCost, self.trueCost, self.sourceFlows,
                            self.sinkFlows, self.arcFlows, self.arcsOpened, "AntColony", False,
                            self.network.isSourceSinkCapacitated, self.network.isSourceSinkCharged)
        return solution

    def initializeAssignedFlowDict(self) -> dict:
        """Adds all possible arcs and supersource/sink as keys to the assigned flow dictionary with a value of zero"""
        assignedFlowDict = {}
        # For all edge, cap pairs, initialize with zero
        for edge in self.network.edgesArray:
            for cap in self.network.possibleArcCapsArray:
                assignedFlowDict[(edge[0], edge[1], cap)] = 0
        # For all supersource -> source and visa versa, initialize with zero
        for srcIndex in range(self.network.numSources):
            source = self.network.sourcesArray[srcIndex]
            cap = self.network.sourceCapsArray[srcIndex]
            assignedFlowDict[(-1, source, cap)] = 0
            assignedFlowDict[(source, -1, -1)] = 0
        # For all supersink -> sink, initialize with zero (NOTE: You can't go back from a supersink)
        for sinkIndex in range(self.network.numSinks):
            sink = self.network.sinksArray[sinkIndex]
            cap = self.network.sinkCapsArray[sinkIndex]
            assignedFlowDict[(sink, -2, cap)] = 0
        return assignedFlowDict

    def initializeAvailableCapacityDict(self) -> dict:
        """Adds all possible arcs and supersource/sink as keys to the available capacity dictionary with value capacity"""
        availableCapacity = {}
        # For all edge, cap pairs, initialize with cap
        for edge in self.network.edgesArray:
            for cap in self.network.possibleArcCapsArray:
                availableCapacity[(edge[0], edge[1], cap)] = cap
        # For all supersource -> source initialize with cap, and for visa versa initialize with MAX_INT
        for srcIndex in range(self.network.numSources):
            source = self.network.sourcesArray[srcIndex]
            cap = self.network.sourceCapsArray[srcIndex]
            availableCapacity[(-1, source, cap)] = cap
            availableCapacity[(source, -1, -1)] = sys.maxsize
        # For all supersink -> sink, initialize with zero (NOTE: You can't go back from a supersink)
        for sinkIndex in range(self.network.numSinks):
            sink = self.network.sinksArray[sinkIndex]
            cap = self.network.sinkCapsArray[sinkIndex]
            availableCapacity[(sink, -2, cap)] = cap
        return availableCapacity

    def resetTripAttributes(self) -> None:
        """Resets the trip attributes after going back to the source"""
        self.currentPosition = -1
        self.tripStack = []
        self.nodesVisitedThisTrip = set()

    def resetTourAndSolutionAttributes(self) -> None:
        """Resets the solution/tour attributes after finding a complete solution"""
        # Reset trip attributes
        self.resetTripAttributes()
        # Reset tour attributes
        self.time = 0
        self.numTrips = 0
        self.remainingFlowToAssign = self.minTargetFlow
        self.assignedFlowDict = self.initializeAssignedFlowDict()
        self.availableCapacityDict = self.initializeAvailableCapacityDict()
        # Reset solution attributes
        self.trueCost = 0.0
        self.sourceFlows = []
        self.sinkFlows = []
        self.arcFlows = {}
        self.arcsOpened = {}

    def printTripData(self) -> None:
        """Prints the data at each time step"""
        print("==================== TRIP ====================")
        print("Trip Number = " + str(self.numTrips))
        print("Trip Stack:")
        print(str(self.tripStack))

    def printTimeStepData(self, arcChoice: tuple) -> None:
        """Prints the data at each time step"""
        print("----------- TIME STEP -----------")
        print("Time = " + str(self.time))
        print("Prev. Node = " + str(arcChoice[0]))
        print("Current Node = " + str(arcChoice[1]))
        print("Is Visited Node? " + str(self.isVisitedNode(arcChoice)))
        print("Trip Stack:")
        print(str(self.tripStack))

    @staticmethod
    def printProbabilityDistribution(rng: float, options: list, cumulativeProbabilities: list) -> None:
        """Prints the cumulative probability distribution generated when an ant considers which edge to chose"""
        percents = []
        for prob in cumulativeProbabilities:
            percent = round(prob * 100, 2)
            percents.append(str(percent) + "%")
        print("RNG = " + str(round(rng * 100, 3)) + "%")
        print("OPTIONS = " + str(options))
        print("CDF = " + str(percents))
