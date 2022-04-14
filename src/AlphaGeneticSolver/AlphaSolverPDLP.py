from numpy import ndarray
from ortools.linear_solver import pywraplp

from src.Network.FlowNetwork import FlowNetwork
from src.Network.Solution import Solution


class AlphaSolverPDLP:
    """Class that solves an alpha-relaxed instance approximately via a PDLP gradient descent solver from Google"""

    def __init__(self, network: FlowNetwork, minTargetFlow: float, isSrcSinkConstrained=True, isSrcSinkCharged=True):
        """Constructor of a AlphaSolverPDLP instance"""
        # Input attributes
        self.network = network
        self.minTargetFlow = minTargetFlow
        self.isSrcSinkConstrained = isSrcSinkConstrained
        self.isSrcSinkCharged = isSrcSinkCharged
        # Solver attributes
        self.solver = pywraplp.Solver.CreateSolver("PDLP")
        self.status = None
        self.isRun = False
        self.trueCost = 0.0
        # Pre-build Model
        self.prebuildVariablesAndConstraints()

    def prebuildVariablesAndConstraints(self) -> None:
        """Builds the decision variables and constraints at initialization (the objective function is updated later)"""
        # =================== DECISION VARIABLES ===================
        # Source and sink decision variables and determines flow from/to each node - Indexed on src/sink matrix
        for s in range(self.network.numSources):
            self.solver.NumVar(0, self.solver.infinity(), "s_" + str(s))
        for t in range(self.network.numSinks):
            self.solver.NumVar(0, self.solver.infinity(), "t_" + str(t))
        # Arc flow variables - Indexed on the arc matrix and determines flow on each arc
        for e in range(self.network.numEdges):
            for cap in range(self.network.numArcCaps):
                self.solver.NumVar(0, self.solver.infinity(), "a_" + str(e) + "_" + str(cap))

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
        self.solver.Add(sum(self.solver.LookupVariable("t_" + str(t)) for t in range(self.network.numSinks)) >=
                        self.minTargetFlow, "minFlow")

        # Edge opening/capacity constraints
        for e in range(self.network.numEdges):
            for cap in range(self.network.numArcCaps):
                capacity = self.network.possibleArcCapsArray[cap]
                varName = "a_" + str(e) + "_" + str(cap)
                self.solver.Add(self.solver.LookupVariable(varName) <= capacity, varName + "_Cap")

        # TODO - Figure out how to enforce the one arc per edge constraint in the alpha-relaxed LP
        """
        if self.isOneArcPerEdge is True:
            for i in range(self.network.numEdges):
                name = "e_" + str(i) + "_OneArcPerEdge"
                self.model.add_constraint(sum(self.arcOpenedVars[(i, j)] for j in range(self.network.numArcCaps)) <= 1,
                                          name)
        """

        # Capacity constraints of sources
        if self.network.isSourceSinkCapacitated is True:
            for s in range(self.network.numSources):
                varName = "s_" + str(s)
                self.solver.Add(self.solver.LookupVariable(varName) <= self.network.sourceCapsArray[s],
                                varName + "_Cap")

        # Capacity constraints of sinks
        if self.network.isSourceSinkCapacitated is True:
            for t in range(self.network.numSinks):
                varName = "t_" + str(t)
                self.solver.Add(self.solver.LookupVariable(varName) <= self.network.sinkCapsArray[t], varName + "_Cap")

        # Conservation of flow constraints
        # Source flow conservation
        for s in range(self.network.numSources):
            source = self.network.sourcesArray[s]
            sourceObj = self.network.nodesDict[source]
            outgoingIndexes = []
            for edge in sourceObj.outgoingEdges:
                outgoingIndexes.append(self.network.edgesDict[edge])
            incomingIndexes = []
            for edge in sourceObj.incomingEdges:
                incomingIndexes.append(self.network.edgesDict[edge])
            varName = "s_" + str(s)
            self.solver.Add(self.solver.LookupVariable(varName) ==
                            sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(c)) for i in
                                outgoingIndexes for c in range(self.network.numArcCaps)) -
                            sum(self.solver.LookupVariable("a_" + str(j) + "_" + str(d)) for j in
                                incomingIndexes for d in range(self.network.numArcCaps)), varName + "_Conserv")
        # Sink flow conservation
        for t in range(self.network.numSinks):
            sink = self.network.sinksArray[t]
            sinkObj = self.network.nodesDict[sink]
            incomingIndexes = []
            for edge in sinkObj.incomingEdges:
                incomingIndexes.append(self.network.edgesDict[edge])
            outgoingIndexes = []
            for edge in sinkObj.outgoingEdges:
                outgoingIndexes.append(self.network.edgesDict[edge])
            varName = "t_" + str(t)
            self.solver.Add(self.solver.LookupVariable(varName) ==
                            sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(c)) for i in
                                incomingIndexes for c in range(self.network.numArcCaps)) -
                            sum(self.solver.LookupVariable("a_" + str(j) + "_" + str(d)) for j in
                                outgoingIndexes for d in range(self.network.numArcCaps)), varName + "_Conserv")
        # Intermediate node flow conservation
        for n in range(self.network.numInterNodes):
            interNode = self.network.interNodesArray[n]
            nodeObj = self.network.nodesDict[interNode]
            incomingIndexes = []
            for edge in nodeObj.incomingEdges:
                incomingIndexes.append(self.network.edgesDict[edge])
            outgoingIndexes = []
            for edge in nodeObj.outgoingEdges:
                outgoingIndexes.append(self.network.edgesDict[edge])
            name = "n_" + str(n) + "_Conserv"
            self.solver.Add(0 == sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(c)) for i in
                                     incomingIndexes for c in range(self.network.numArcCaps)) -
                            sum(self.solver.LookupVariable("a_" + str(j) + "_" + str(d)) for j in
                                outgoingIndexes for d in range(self.network.numArcCaps)), name)

    def updateObjectiveFunction(self, alphaValues: ndarray) -> None:
        """Updates the objective function based on the input alpha values"""
        # Clear any existing objective function
        self.solver.Objective().Clear()
        # Write new objective function
        if self.network.isSourceSinkCharged is True:
            self.solver.Minimize(
                sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(j)) * (
                        self.network.getArcVariableCostFromEdgeCapIndices(i, j) +
                        self.network.getArcFixedCostFromEdgeCapIndices(i, j) * alphaValues[i][j]) for i in
                    range(self.network.numEdges)
                    for j in range(self.network.numArcCaps)) + sum(
                    self.solver.LookupVariable("s_" + str(s)) * self.network.sourceVariableCostsArray[s]
                    for s in range(self.network.numSources)) + sum(
                    self.solver.LookupVariable("t_" + str(t)) * self.network.sinkVariableCostsArray[t]
                    for t in range(self.network.numSinks)))
        elif self.network.isSourceSinkCharged is False:
            self.solver.Minimize(
                sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(j)) * (
                        self.network.getArcVariableCostFromEdgeCapIndices(i, j) +
                        self.network.getArcFixedCostFromEdgeCapIndices(i, j) * alphaValues[i][j]) for i
                    in range(self.network.numEdges)
                    for j in range(self.network.numArcCaps)))

    def solveModel(self) -> None:
        """Solves the alpha-relaxed LP model with PDLP"""
        # print("\nAttempting to solve model...")
        self.status = self.solver.Solve()
        self.isRun = True
        # print("Solver execution complete...\n")

    def writeSolution(self) -> Solution:
        """Saves the solution instance"""
        if self.isRun is False:
            print("You must run the solver before building a solution!")
        elif self.status == pywraplp.Solver.OPTIMAL:
            print("Building solution...")
            objValue = self.solver.Objective().Value()
            srcFlows = self.getSrcFlowsList()
            sinkFlows = self.getSinkFlowsList()
            arcFlows = self.getArcFlowsDict()
            arcsOpen = self.getArcsOpenDict()
            self.trueCost = self.calculateTrueCost()
            thisSolution = Solution(self.network, self.minTargetFlow, objValue, self.trueCost, srcFlows, sinkFlows,
                                    arcFlows,
                                    arcsOpen, "gor_PDLP", False, self.isSrcSinkConstrained, self.isSrcSinkCharged)
            print("Solution built!")
            return thisSolution
        else:
            print("No feasible solution exists!")

    def resetSolver(self) -> None:
        """Resets all the output data structures of the solver (but model variables and constraints remain)"""
        self.status = None
        self.isRun = False
        self.trueCost = 0.0

    def calculateTrueCost(self) -> float:
        """Calculates the true cost of the alpha-relaxed LP's output with the true discrete FCNF objective function"""
        srcFlows = self.getSrcFlowsList()
        sinkFlows = self.getSinkFlowsList()
        arcFlows = self.getArcFlowsDict()
        arcsOpen = self.getArcsOpenDict()
        trueCost = 0.0
        if self.isSrcSinkCharged is True:
            for s in range(self.network.numSources):
                trueCost += self.network.sourceVariableCostsArray[s] * srcFlows[s]
            for t in range(self.network.numSinks):
                trueCost += self.network.sinkVariableCostsArray[t] * sinkFlows[t]
        for edge in range(self.network.numEdges):
            for cap in range(self.network.numArcCaps):
                if arcsOpen[(edge, cap)] == 1:
                    arcVariableCost = self.network.getArcVariableCostFromEdgeCapIndices(edge, cap)
                    arcFixedCost = self.network.getArcFixedCostFromEdgeCapIndices(edge, cap)
                    trueCost += arcVariableCost * arcFlows[(edge, cap)] + arcFixedCost
        return trueCost

    def getObjectiveValue(self) -> float:
        """Returns the objective value (i.e. fake cost) of the alpha-relaxed LP solver"""
        return self.solver.Objective().Value()

    def getArcFlowsDict(self) -> dict:
        """Returns the dictionary of arc flows with key (edgeIndex, capIndex)"""
        arcFlows = {}
        for edge in range(self.network.numEdges):
            for cap in range(self.network.numArcCaps):
                thisFlow = self.solver.LookupVariable("a_" + str(edge) + "_" + str(cap)).SolutionValue()
                arcFlows[(edge, cap)] = thisFlow
        return arcFlows

    def getArcsOpenDict(self) -> dict:
        """Returns the dictionary of arcs opened with key (edgeIndex, capIndex)"""
        arcsOpen = {}
        for edge in range(self.network.numEdges):
            for cap in range(self.network.numArcCaps):
                thisFlow = self.solver.LookupVariable("a_" + str(edge) + "_" + str(cap)).SolutionValue()
                if thisFlow > 0:
                    arcsOpen[(edge, cap)] = 1
                else:
                    arcsOpen[(edge, cap)] = 0
        return arcsOpen

    def getSrcFlowsList(self) -> list:
        """Returns the list of source flows"""
        srcFlows = []
        for s in range(self.network.numSources):
            srcFlows.append(self.solver.LookupVariable("s_" + str(s)).SolutionValue())
        return srcFlows

    def getSinkFlowsList(self) -> list:
        """Returns the list of sink flows"""
        sinkFlows = []
        for t in range(self.network.numSinks):
            sinkFlows.append(self.solver.LookupVariable("t_" + str(t)).SolutionValue())
        return sinkFlows

    def printSolverOverview(self) -> None:
        """Prints the most important and concise details of the solver, model and solution"""
        if self.status == pywraplp.Solver.OPTIMAL:
            print("Solution:")
            print("Number of Variables = " + str(self.solver.NumVariables()))
            print("Number of Constraints = " + str(self.solver.NumConstraints()))
            print("Objective Value of Relaxed LP (i.e. Fake Cost) = " + str(round(self.solver.Objective().Value(), 1)))
            print("Objective Value  if FCNF (i.e. True Cost) = " + str(round(self.trueCost, 1)))
            print("Solved by: Google OR's PDLP Gradient Descent Solver\n")
