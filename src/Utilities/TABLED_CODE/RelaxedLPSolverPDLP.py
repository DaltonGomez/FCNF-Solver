from numpy import ndarray
from ortools.linear_solver import pywraplp

from src.FlowNetwork.FlowNetworkSolution import FlowNetworkSolution
from src.Graph.CandidateGraph import CandidateGraph


class RelaxedLPSolverPDLP:
    """Class that solves an alpha-relaxed instance approximately via a PDLP gradient descent solver from Google OR-Tools"""

    def __init__(self, graph: CandidateGraph, minTargetFlow: float, isSourceSinkCapacitated=True, isSourceSinkCharged=False):
        """Constructor of a AlphaSolverPDLP instance"""
        # Input attributes
        self.graph: CandidateGraph = graph  # Input candidate graph to solve optimally
        self.minTargetFlow: float = minTargetFlow  # Target flow that the solution must capture
        self.isSourceSinkCapacitated: bool = isSourceSinkCapacitated  # Boolean indicating if the input graph contained src/sink capacities, which were considered by the solver
        self.isSourceSinkCharged: bool = isSourceSinkCharged  # Boolean indicating if the input graph contained src/sink charges, which were considered by the solver
        # Solver attributes
        self.solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("PDLP")  # Solver object acting as a wrapper to Google OR-Tools PDLP solver
        self.status = None  # Captures the returned value of solving the solver object
        self.isRun = False  # Boolean indicating if the solver has been run
        self.trueCost: float = 0.0  # True cost of the solution under the Fixed-Charge Network Flow model
        # Pre-build Model
        self.prebuildVariablesAndConstraints()  # Called on initialization to build only the variables and constraints in the model

    def prebuildVariablesAndConstraints(self) -> None:
        """Builds the decision variables and constraints at initialization (the objective function is updated later)"""
        # =================== DECISION VARIABLES ===================
        # Source and sink decision variables and determines flow from/to each node - Indexed on src/sink matrix
        for s in range(self.graph.numSources):
            self.solver.NumVar(0, self.solver.infinity(), "s_" + str(s))
        for t in range(self.graph.numSinks):
            self.solver.NumVar(0, self.solver.infinity(), "t_" + str(t))
        # Arc flow variables - Indexed on the arc matrix and determines flow on each arc
        for e in range(self.graph.numEdges):
            for cap in range(self.graph.numArcsPerEdge):
                self.solver.NumVar(0, self.solver.infinity(), "a_" + str(e) + "_" + str(cap))

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
        self.solver.Add(sum(self.solver.LookupVariable("t_" + str(t)) for t in range(self.graph.numSinks)) >=
                        self.minTargetFlow, "minFlow")

        # Edge opening/capacity constraints
        for e in range(self.graph.numEdges):
            for cap in range(self.graph.numArcsPerEdge):
                capacity = self.graph.possibleArcCapsArray[cap]
                varName = "a_" + str(e) + "_" + str(cap)
                self.solver.Add(self.solver.LookupVariable(varName) <= capacity, varName + "_Cap")

        # Capacity constraints of sources
        if self.graph.isSourceSinkCapacitated is True:
            for s in range(self.graph.numSources):
                varName = "s_" + str(s)
                self.solver.Add(self.solver.LookupVariable(varName) <= self.graph.sourceCapsArray[s],
                                varName + "_Cap")

        # Capacity constraints of sinks
        if self.graph.isSourceSinkCapacitated is True:
            for t in range(self.graph.numSinks):
                varName = "t_" + str(t)
                self.solver.Add(self.solver.LookupVariable(varName) <= self.graph.sinkCapsArray[t], varName + "_Cap")

        # Conservation of flow constraints
        # Source flow conservation
        for s in range(self.graph.numSources):
            source = self.graph.sourcesArray[s]
            sourceObj = self.graph.nodesDict[source]
            outgoingIndexes = []
            for edge in sourceObj.outgoingEdges:
                outgoingIndexes.append(self.graph.edgesDict[edge])
            incomingIndexes = []
            for edge in sourceObj.incomingEdges:
                incomingIndexes.append(self.graph.edgesDict[edge])
            varName = "s_" + str(s)
            self.solver.Add(self.solver.LookupVariable(varName) ==
                            sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(c)) for i in
                                outgoingIndexes for c in range(self.graph.numArcsPerEdge)) -
                            sum(self.solver.LookupVariable("a_" + str(j) + "_" + str(d)) for j in
                                incomingIndexes for d in range(self.graph.numArcsPerEdge)), varName + "_Conserv")
        # Sink flow conservation
        for t in range(self.graph.numSinks):
            sink = self.graph.sinksArray[t]
            sinkObj = self.graph.nodesDict[sink]
            incomingIndexes = []
            for edge in sinkObj.incomingEdges:
                incomingIndexes.append(self.graph.edgesDict[edge])
            outgoingIndexes = []
            for edge in sinkObj.outgoingEdges:
                outgoingIndexes.append(self.graph.edgesDict[edge])
            varName = "t_" + str(t)
            self.solver.Add(self.solver.LookupVariable(varName) ==
                            sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(c)) for i in
                                incomingIndexes for c in range(self.graph.numArcsPerEdge)) -
                            sum(self.solver.LookupVariable("a_" + str(j) + "_" + str(d)) for j in
                                outgoingIndexes for d in range(self.graph.numArcsPerEdge)), varName + "_Conserv")
        # Intermediate node flow conservation
        for n in range(self.graph.numInterNodes):
            interNode = self.graph.interNodesArray[n]
            nodeObj = self.graph.nodesDict[interNode]
            incomingIndexes = []
            for edge in nodeObj.incomingEdges:
                incomingIndexes.append(self.graph.edgesDict[edge])
            outgoingIndexes = []
            for edge in nodeObj.outgoingEdges:
                outgoingIndexes.append(self.graph.edgesDict[edge])
            name = "n_" + str(n) + "_Conserv"
            self.solver.Add(0 == sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(c))
                                     for i in incomingIndexes for c in range(self.graph.numArcsPerEdge)) -
                                sum(self.solver.LookupVariable("a_" + str(j) + "_" + str(d))
                                    for j in outgoingIndexes for d in range(self.graph.numArcsPerEdge)), name)

    def updateObjectiveFunction(self, alphaValues: ndarray) -> None:
        """Updates the objective function based on the input alpha values"""
        # Clear any existing objective function
        self.solver.Objective().Clear()
        # Write new objective function
        if self.graph.isSourceSinkCharged is True:
            self.solver.Minimize(
                sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(j)) * (self.graph.getArcVariableCostFromEdgeCapIndices(i, j) +
                    self.graph.getArcFixedCostFromEdgeCapIndices(i, j) * alphaValues[i][j])
                    for i in range(self.graph.numEdges)
                    for j in range(self.graph.numArcsPerEdge)) +
                sum(self.solver.LookupVariable("s_" + str(s)) * self.graph.sourceVariableCostsArray[s]
                    for s in range(self.graph.numSources)) +
                sum(self.solver.LookupVariable("t_" + str(t)) * self.graph.sinkVariableCostsArray[t]
                    for t in range(self.graph.numSinks)))
        elif self.graph.isSourceSinkCharged is False:
            self.solver.Minimize(
                sum(self.solver.LookupVariable("a_" + str(i) + "_" + str(j)) * (self.graph.getArcVariableCostFromEdgeCapIndices(i, j) +
                    self.graph.getArcFixedCostFromEdgeCapIndices(i, j) * alphaValues[i][j])
                    for i in range(self.graph.numEdges)
                    for j in range(self.graph.numArcsPerEdge)))

    def solveModel(self) -> None:
        """Solves the alpha-relaxed LP model with PDLP"""
        # print("\nAttempting to solve model...")
        self.status = self.solver.Solve()
        self.isRun = True
        # print("Solver execution complete...\n")

    def writeSolution(self) -> FlowNetworkSolution:
        """Saves the solution instance"""
        if self.isRun is False:
            print("You must run the solver before building a solution!")
        elif self.status == pywraplp.Solver.OPTIMAL:
            # print("Building solution...")  # PRINT OPTION
            objValue = self.solver.Objective().Value()
            srcFlows = self.getSrcFlowsList()
            sinkFlows = self.getSinkFlowsList()
            arcFlows = self.getArcFlowsDict()
            self.trueCost = self.calculateTrueCost()
            thisSolution = FlowNetworkSolution(self.graph, self.minTargetFlow, objValue, self.trueCost,
                                               srcFlows, sinkFlows, arcFlows, "gor_PDLP", False,
                                               self.isSourceSinkCapacitated, self.isSourceSinkCharged)
            # print("Solution built!")  # PRINT OPTION
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
        trueCost = 0.0
        if self.isSourceSinkCharged is True:
            for s in range(self.graph.numSources):
                trueCost += self.graph.sourceVariableCostsArray[s] * srcFlows[s]
            for t in range(self.graph.numSinks):
                trueCost += self.graph.sinkVariableCostsArray[t] * sinkFlows[t]
        for edge in range(self.graph.numEdges):
            for cap in range(self.graph.numArcsPerEdge):
                if arcFlows[(edge, cap)] > 0.0:
                    arcVariableCost = self.graph.getArcVariableCostFromEdgeCapIndices(edge, cap)
                    arcFixedCost = self.graph.getArcFixedCostFromEdgeCapIndices(edge, cap)
                    trueCost += arcVariableCost * arcFlows[(edge, cap)] + arcFixedCost
        return trueCost

    def getObjectiveValue(self) -> float:
        """Returns the objective value (i.e. fake cost) of the alpha-relaxed LP solver"""
        return self.solver.Objective().Value()

    def getArcFlowsDict(self) -> dict:
        """Returns the dictionary of arc flows with key (edgeIndex, capIndex)"""
        arcFlows = {}
        for edge in range(self.graph.numEdges):
            for cap in range(self.graph.numArcsPerEdge):
                thisFlow = self.solver.LookupVariable("a_" + str(edge) + "_" + str(cap)).SolutionValue()
                arcFlows[(edge, cap)] = thisFlow
        return arcFlows

    def getSrcFlowsList(self) -> list:
        """Returns the list of source flows"""
        srcFlows = []
        for s in range(self.graph.numSources):
            srcFlows.append(self.solver.LookupVariable("s_" + str(s)).SolutionValue())
        return srcFlows

    def getSinkFlowsList(self) -> list:
        """Returns the list of sink flows"""
        sinkFlows = []
        for t in range(self.graph.numSinks):
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