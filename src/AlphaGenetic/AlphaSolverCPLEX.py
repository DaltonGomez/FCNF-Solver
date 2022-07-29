from typing import List, Dict, Tuple

from docplex.mp.model import Model
from numpy import ndarray

from src.FlowNetwork.FlowNetworkSolution import FlowNetworkSolution
from src.Graph.CandidateGraph import CandidateGraph


class AlphaSolverCPLEX:
    """Class that solves a candidate graph instance approximately via an alpha-relaxed LP model within CPLEX"""

    def __init__(self, graph: CandidateGraph, minTargetFlow: float, isOneDimAlphaTable=True,
                 isOptimizedArcSelections=True, logOutput=False):
        """Constructor of a AlphaSolverCPLEX instance"""
        # Input attributes
        self.graph: CandidateGraph = graph  # Input candidate graph to solve optimally
        self.minTargetFlow: float = minTargetFlow  # Target flow that the solution must capture
        self.isSourceSinkCapacitated: bool = self.graph.isSourceSinkCapacitated  # Boolean indicating if the input graph contained src/sink capacities, which were considered by the solver
        self.isSourceSinkCharged: bool = self.graph.isSourceSinkCharged  # Boolean indicating if the input graph contained src/sink charges, which were considered by the solver
        self.isOneDimAlphaTable: bool = isOneDimAlphaTable  # Boolean indicating if the alpha table is only one dimensional (i.e. only one arc per edge)
        # Solver model
        self.solver: Model = Model(name="Alpha-LP-FCNF-Solver", log_output=logOutput,
                                   cts_by_name=True)  # Model object acting as a wrapper to local CPLEX installation
        self.isRun: bool = False  # Boolean indicating if the solver has been run
        self.isOptimizedArcSelections: bool = isOptimizedArcSelections  # Boolean indicating if the optimal arc should be selected for the assigned flow
        self.trueCost: float = 0.0  # True cost of the solution under the Fixed-Charge Network Flow model
        # Decision variables
        self.sourceFlowVars: List[float] = []  # List of the flow values assigned to each source, indexed the same as the graph.sourcesArray
        self.sinkFlowVars: List[float] = []  # List of the flow values assigned to each sink, indexed the same as the graph.sinksArray
        self.arcFlowVars: Dict[Tuple[int, int], float] = {}  # Dictionary mapping (edgeIndex, arcIndex) keys to assigned flow values
        # Pre-build Model
        self.prebuildVariablesAndConstraints()  # Called on initialization to build only the variables and constraints in the model

    def prebuildVariablesAndConstraints(self) -> None:
        """Builds the decision variables and constraints at initialization (the objective function is updated later)"""
        if self.isOneDimAlphaTable is False:
            # =================== DECISION VARIABLES ===================
            # Source and sink decision variables and determines flow from/to each node - Indexed on src/sink matrix
            self.sourceFlowVars = self.solver.continuous_var_list(self.graph.numSources, name="s", lb=0)
            self.sinkFlowVars = self.solver.continuous_var_list(self.graph.numSinks, name="t", lb=0)
            # Arc decision variables - Indexed on the arc matrix and determines flow on each arc
            self.arcFlowVars = self.solver.continuous_var_matrix(self.graph.numEdges, self.graph.numArcsPerEdge,
                                                                 name="a", lb=0)

            # =================== CONSTRAINTS ===================
            # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
            self.solver.add_constraint(
                sum(self.sinkFlowVars[i] for i in range(self.graph.numSinks)) >= self.minTargetFlow,
                ctname="minFlow")

            # Edge opening/capacity constraints
            for i in range(self.graph.numEdges):
                for j in range(self.graph.numArcsPerEdge):
                    capacity = self.graph.possibleArcCapsArray[j]
                    arcID = (self.graph.edgesArray[i][0], self.graph.edgesArray[i][1], capacity)
                    ctName = "a_" + str(arcID) + "_Cap"
                    self.solver.add_constraint(self.arcFlowVars[(i, j)] <= capacity, ctname=ctName)

            # Capacity constraints of sources
            if self.graph.isSourceSinkCapacitated is True:
                for i in range(self.graph.numSources):
                    ctName = "s_" + str(self.graph.sourcesArray[i]) + "_Cap"
                    self.solver.add_constraint(self.sourceFlowVars[i] <= self.graph.sourceCapsArray[i], ctname=ctName)

            # Capacity constraints of sinks
            if self.graph.isSourceSinkCapacitated is True:
                for i in range(self.graph.numSinks):
                    ctName = "t_" + str(self.graph.sinksArray[i]) + "_Cap"
                    self.solver.add_constraint(self.sinkFlowVars[i] <= self.graph.sinkCapsArray[i], ctname=ctName)

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
                ctName = "s_" + str(source) + "_Conserv"
                self.solver.add_constraint(self.sourceFlowVars[s] ==
                                           sum(self.arcFlowVars[(m, n)]
                                               for m in outgoingIndexes
                                               for n in range(self.graph.numArcsPerEdge)) -
                                           sum(self.arcFlowVars[(i, j)]
                                               for i in incomingIndexes
                                               for j in range(self.graph.numArcsPerEdge)), ctName)

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
                ctName = "t_" + str(sink) + "_Conserv"
                self.solver.add_constraint(self.sinkFlowVars[t] ==
                                           sum(self.arcFlowVars[(m, n)]
                                               for m in incomingIndexes
                                               for n in range(self.graph.numArcsPerEdge)) -
                                           sum(self.arcFlowVars[(i, j)]
                                               for i in outgoingIndexes
                                               for j in range(self.graph.numArcsPerEdge)), ctName)

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
                ctName = "n_" + str(interNode) + "_Conserv"
                self.solver.add_constraint(sum(self.arcFlowVars[(i, j)]
                                               for i in incomingIndexes
                                               for j in range(self.graph.numArcsPerEdge)) -
                                           sum(self.arcFlowVars[(m, n)]
                                               for m in outgoingIndexes
                                               for n in range(self.graph.numArcsPerEdge)) == 0, ctname=ctName)
        elif self.isOneDimAlphaTable is True:
            # =================== DECISION VARIABLES ===================
            # Source and sink decision variables and determines flow from/to each node - Indexed on src/sink matrix
            self.sourceFlowVars = self.solver.continuous_var_list(self.graph.numSources, name="s", lb=0)
            self.sinkFlowVars = self.solver.continuous_var_list(self.graph.numSinks, name="t", lb=0)
            # Arc decision variables - Indexed on the arc matrix and determines flow on each arc
            self.arcFlowVars = self.solver.continuous_var_matrix(self.graph.numEdges, 1, name="a", lb=0)

            # =================== CONSTRAINTS ===================
            # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
            self.solver.add_constraint(
                sum(self.sinkFlowVars[i] for i in range(self.graph.numSinks)) >= self.minTargetFlow,
                ctname="minFlow")

            # Edge opening/capacity constraints
            for i in range(self.graph.numEdges):
                capacity = self.graph.possibleArcCapsArray[-1]
                arcID = (self.graph.edgesArray[i][0], self.graph.edgesArray[i][1], capacity)
                ctName = "a_" + str(arcID) + "_Cap"
                self.solver.add_constraint(self.arcFlowVars[(i, 0)] <= capacity, ctname=ctName)

            # Capacity constraints of sources
            if self.graph.isSourceSinkCapacitated is True:
                for i in range(self.graph.numSources):
                    ctName = "s_" + str(self.graph.sourcesArray[i]) + "_Cap"
                    self.solver.add_constraint(self.sourceFlowVars[i] <= self.graph.sourceCapsArray[i], ctname=ctName)

            # Capacity constraints of sinks
            if self.graph.isSourceSinkCapacitated is True:
                for i in range(self.graph.numSinks):
                    ctName = "t_" + str(self.graph.sinksArray[i]) + "_Cap"
                    self.solver.add_constraint(self.sinkFlowVars[i] <= self.graph.sinkCapsArray[i], ctname=ctName)

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
                ctName = "s_" + str(source) + "_Conserv"
                self.solver.add_constraint(self.sourceFlowVars[s] ==
                                           sum(self.arcFlowVars[(m, 0)]
                                               for m in outgoingIndexes) -
                                           sum(self.arcFlowVars[(i, 0)]
                                               for i in incomingIndexes), ctName)

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
                ctName = "t_" + str(sink) + "_Conserv"
                self.solver.add_constraint(self.sinkFlowVars[t] ==
                                           sum(self.arcFlowVars[(m, 0)]
                                               for m in incomingIndexes) -
                                           sum(self.arcFlowVars[(i, 0)]
                                               for i in outgoingIndexes), ctName)

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
                ctName = "n_" + str(interNode) + "_Conserv"
                self.solver.add_constraint(sum(self.arcFlowVars[(i, 0)]
                                               for i in incomingIndexes) -
                                           sum(self.arcFlowVars[(m, 0)]
                                               for m in outgoingIndexes) == 0, ctname=ctName)

    def updateObjectiveFunction(self, alphaValues: ndarray) -> None:
        """Updates the objective function based on the input alpha values"""
        # Clear any existing objective function
        self.solver.remove_objective()
        # Model if NOT reducing to a one-dimensional alpha table
        if self.isOneDimAlphaTable is False:
            if self.graph.isSourceSinkCharged is True:
                self.solver.set_objective("min", sum(
                    self.arcFlowVars[(i, j)] *
                    (self.graph.getArcVariableCostFromEdgeCapIndices(i, j) +
                     self.graph.getArcFixedCostFromEdgeCapIndices(i, j) *
                     alphaValues[i][j])
                    for i in range(self.graph.numEdges)
                    for j in range(self.graph.numArcsPerEdge)) +
                                          sum(self.sourceFlowVars[s] * self.graph.sourceVariableCostsArray[s]
                                              for s in range(self.graph.numSources)) +
                                          sum(self.sinkFlowVars[t] * self.graph.sinkVariableCostsArray[t]
                                              for t in range(self.graph.numSinks)))
            elif self.graph.isSourceSinkCharged is False:
                self.solver.set_objective("min", sum(
                    self.arcFlowVars[(i, j)] *
                    (self.graph.getArcVariableCostFromEdgeCapIndices(i, j) +
                     self.graph.getArcFixedCostFromEdgeCapIndices(i, j) *
                     alphaValues[i][j])
                    for i in range(self.graph.numEdges)
                    for j in range(self.graph.numArcsPerEdge)))
        elif self.isOneDimAlphaTable is True:
            if self.graph.isSourceSinkCharged is True:
                self.solver.set_objective("min", sum(
                    self.arcFlowVars[(i, 0)] *
                    (self.graph.getArcVariableCostFromEdgeCapIndices(i, self.graph.numArcsPerEdge - 1) +
                     self.graph.getArcFixedCostFromEdgeCapIndices(i, self.graph.numArcsPerEdge - 1) *
                     alphaValues[i][self.graph.numArcsPerEdge - 1])
                    for i in range(self.graph.numEdges)) +
                                          sum(self.sourceFlowVars[s] * self.graph.sourceVariableCostsArray[s]
                                              for s in range(self.graph.numSources)) +
                                          sum(self.sinkFlowVars[t] * self.graph.sinkVariableCostsArray[t]
                                              for t in range(self.graph.numSinks)))
            elif self.graph.isSourceSinkCharged is False:
                self.solver.set_objective("min", sum(
                    self.arcFlowVars[(i, 0)] *
                    (self.graph.getArcVariableCostFromEdgeCapIndices(i, self.graph.numArcsPerEdge - 1) +
                     self.graph.getArcFixedCostFromEdgeCapIndices(i, self.graph.numArcsPerEdge - 1) *
                     alphaValues[i][self.graph.numArcsPerEdge - 1])
                    for i in range(self.graph.numEdges)))

    def solveModel(self) -> None:
        """Solves the alpha-relaxed LP model in CPLEX"""
        self.solver.solve()
        self.isRun = True

    def getArcFlowsDict(self) -> dict:
        """Returns the dictionary of arc flows with key (edgeIndex, capIndex)"""
        if self.isOneDimAlphaTable is False:
            arcFlows = self.solver.solution.get_value_dict(self.arcFlowVars)
        else:
            arcFlows = {}
            solverFlows = self.solver.solution.get_value_dict(self.arcFlowVars)
            largestCapIndex = self.graph.numArcsPerEdge - 1
            for edge in range(self.graph.numEdges):
                for cap in range(self.graph.numArcsPerEdge):
                    # Initialize all dictionary values with zero
                    arcFlows[(edge, cap)] = 0
                # Update maximum capacity value for this edge
                thisFlow = solverFlows[(edge, 0)]
                arcFlows[(edge, largestCapIndex)] = thisFlow
        return arcFlows

    def getSrcFlowsList(self) -> list:
        """Returns the list of source flows"""
        srcFlows = self.solver.solution.get_value_list(self.sourceFlowVars)
        return srcFlows

    def getSinkFlowsList(self) -> list:
        """Returns the list of sink flows"""
        sinkFlows = self.solver.solution.get_value_list(self.sinkFlowVars)
        return sinkFlows

    def optimizeArcSelection(self, rawArcFlows: dict) -> dict:
        """Iterates over all opened edges and picks the arc capacity that best fits the assigned flow"""
        optimalArcFlows = {}
        # Iterate over all edges in the candidate graph
        for edgeIndex in range(self.graph.numEdges):
            thisEdgeFlow = 0.0
            # Iterate over all arcs and compute total assigned flow on the edge
            for arcIndex in range(self.graph.numArcsPerEdge):
                thisEdgeFlow += rawArcFlows[(edgeIndex, arcIndex)]
            # Determine the optimal arc capacity to assign the edge
            optimalCapIndex = self.getOptimalArcCapIndex(thisEdgeFlow)
            # Iterate back over arcs, only opening/assigning flow to the optimal arc; close all others
            for arcIndex in range(self.graph.numArcsPerEdge):
                arcKeyTuple = (edgeIndex, arcIndex)
                if arcIndex == optimalCapIndex and thisEdgeFlow > 0.0:
                    optimalArcFlows[arcKeyTuple] = thisEdgeFlow
                else:
                    optimalArcFlows[arcKeyTuple] = 0.0
        return optimalArcFlows

    def getOptimalArcCapIndex(self, totalAssignedFlow: float) -> int:
        """Returns the optimal arc capacity for the edge based on the total assigned flow"""
        for arcCapIndex in range(self.graph.numArcsPerEdge):
            if self.graph.possibleArcCapsArray[arcCapIndex] >= totalAssignedFlow:
                return arcCapIndex

    def calculateTrueCost(self) -> float:
        """Calculates the true cost of the alpha-relaxed LP's output with the true discrete FCNF objective function"""
        srcFlows = self.getSrcFlowsList()
        sinkFlows = self.getSinkFlowsList()
        arcFlows = self.getArcFlowsDict()
        if self.isOptimizedArcSelections is True:
            arcFlows = self.optimizeArcSelection(arcFlows)
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
        """Returns the objective value found by the CPLEX relaxed-LP solver"""
        return self.solver.solution.get_objective_value()

    def writeSolution(self) -> FlowNetworkSolution:
        """Writes out the solution instance"""
        if self.isRun is False:
            print("You must run the solver before building a solution!")
        elif self.solver.solution is not None:
            # print("Building solution...")  # PRINT OPTION
            objValue = self.getObjectiveValue()
            srcFlows = self.getSrcFlowsList()
            sinkFlows = self.getSinkFlowsList()
            arcFlows = self.getArcFlowsDict()
            if self.isOptimizedArcSelections is True:
                arcFlows = self.optimizeArcSelection(arcFlows)
            self.trueCost = self.calculateTrueCost()
            thisSolution = FlowNetworkSolution(self.graph, self.minTargetFlow, objValue, self.trueCost,
                                               srcFlows, sinkFlows, arcFlows, "alphaLP_CPLEX", False,
                                               self.isSourceSinkCapacitated, self.isSourceSinkCharged,
                                               optionalDescription="1D_Alpha_Table = " + str(self.isOneDimAlphaTable))
            print("Solution built!")
            return thisSolution
        else:
            print("No feasible solution exists!")

    def resetSolver(self) -> None:
        """Resets all the output data structures of the solver (but model variables and constraints remain)"""
        self.isRun = False
        self.trueCost = 0.0

    def setTimeLimit(self, timeLimitInSeconds: float) -> None:
        """Sets the time limit, in seconds, for the CPLEX relaxed-LP model to run"""
        print("Setting CPLEX time limit to " + str(timeLimitInSeconds) + " seconds...")
        self.solver.set_time_limit(timeLimitInSeconds)

    def getTimeLimit(self) -> float:
        """Gets the time limit, in seconds, for the CPLEX relaxed-LP model to run"""
        return self.solver.get_time_limit()

    def getCplexStatus(self) -> str:
        """Returns the status of the solution found by the CPLEX relaxed-LP solver"""
        return self.solver.solve_details.status

    def getCplexStatusCode(self) -> int:
        """Returns the status code of the solution found by the CPLEX relaxed-LP solver. See the following for mapping codes:
        https://www.ibm.com/docs/en/icos/20.1.0?topic=micclcarm-solution-status-codes-by-number-in-cplex-callable-library-c-api"""
        return self.solver.solve_details.status_code

    def getCplexRuntime(self) -> float:
        """Returns the runtime, in seconds, of the CPLEX relaxed-LP solver"""
        return self.solver.solve_details.time

    def getDeterministicTime(self) -> float:
        """Returns the deterministic time (i.e. number of ticks) of the CPLEX MILP solver"""
        return self.solver.solve_details.deterministic_time

    def printAllSolverData(self) -> None:
        """Prints all the data store within the solver class"""
        self.printSolverOverview()
        self.printModel()
        self.printSolution()

    def printSolverOverview(self) -> None:
        """Prints the most important and concise details of the solver, model and solution"""
        self.solver.print_information()
        if self.isRun is True:
            print(self.solver.get_solve_details())
            print("Solved by= " + self.solver.solution.solved_by + "\n")
            self.solver.print_solution()

    def printModel(self) -> None:
        """Prints all constraints of the MILP model for the FCNF instance (FOR DEBUGGING-DON'T CALL ON LARGE INPUTS)"""
        print("=============== MILP MODEL OF FCNF INSTANCE ========================")
        self.solver.print_information()
        print("=============== OBJECTIVE FUNCTION ========================")
        print(self.solver.get_objective_expr())
        print("=============== CONSTRAINTS ========================")
        print(self.solver.get_constraint_by_name("minFlow"))
        for s in self.graph.sourcesArray:
            print(self.solver.get_constraint_by_name("s_" + str(s) + "_Conserv"))
            print(self.solver.get_constraint_by_name("s_" + str(s) + "_Cap"))
        for t in self.graph.sinksArray:
            print(self.solver.get_constraint_by_name("t_" + str(t) + "_Conserv"))
            print(self.solver.get_constraint_by_name("t_" + str(t) + "_Cap"))
        for n in self.graph.interNodesArray:
            print(self.solver.get_constraint_by_name("n_" + str(n) + "_Conserv"))
        for i in range(self.graph.numEdges):
            for j in range(self.graph.numArcsPerEdge):
                capacity = self.graph.possibleArcCapsArray[j]
                arcID = (self.graph.edgesArray[i][0], self.graph.edgesArray[i][1], capacity)
                print(self.solver.get_constraint_by_name("a_" + str(arcID) + "_Cap"))

    def printSolution(self) -> None:
        """Prints the solution data of the FCNF instance solved by the MILP model"""
        print("=============== SOLUTION DETAILS ========================")
        print(self.solver.get_solve_details())
        if self.isRun is True:
            print("Solved by= " + self.solver.solution.solved_by + "\n")
            print("=============== SOLUTION VALUES ========================")
            self.solver.print_solution()
        else:
            print("No feasible solution exists!")
