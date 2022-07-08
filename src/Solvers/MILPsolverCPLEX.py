from typing import List, Dict, Tuple

from docplex.mp.model import Model
from docplex.mp.progress import ProgressDataRecorder

from src.FlowNetwork.FlowNetworkSolution import FlowNetworkSolution
from src.Graph.CandidateGraph import CandidateGraph


class MILPsolverCPLEX:
    """Class that solves a candidate graph instance optimally via a MILP model within CPLEX"""

    def __init__(self, graph: CandidateGraph, minTargetFlow: float, isOneArcPerEdge=False, logOutput=False):
        """Constructor of a MILPsolverCPLEX instance"""
        # Input attributes
        self.graph: CandidateGraph = graph  # Input candidate graph to solve optimally
        self.minTargetFlow: float = minTargetFlow  # Target flow that the solution must capture
        self.isOneArcPerEdge: bool = isOneArcPerEdge  # Boolean indicating if the solver considered the constraint that only opens one arc per edge (MILP only)
        self.isSourceSinkCapacitated: bool = self.graph.isSourceSinkCapacitated  # Boolean indicating if the input graph contained src/sink capacities, which were considered by the solver
        self.isSourceSinkCharged: bool = self.graph.isSourceSinkCharged  # Boolean indicating if the input graph contained src/sink charges, which were considered by the solver
        # Solver model
        self.model: Model = Model(name="FCNF-MILP-Solvers", log_output=logOutput, cts_by_name=True)  # Model object acting as a wrapper to local CPLEX installation
        self.progressDataRecorder: ProgressDataRecorder = ProgressDataRecorder(clock="gap")  # Create a progress listener for the MILP model in CPLEX
        self.model.add_progress_listener(self.progressDataRecorder)  # Attach the progress listener to the MILP model
        self.runtimeTimestamps: List = []  # Stores the runtime timestamps, in seconds, from CPLEX while solving the MILP
        self.runtimeObjectiveValues: List = []  # Stores the runtime objective values from CPLEX while solving the MILP
        self.runtimeBestBounds: List = []  # Stores the runtime best bounds from CPLEX while solving the MILP
        self.runtimeGaps: List = []  # Stores the runtime gaps of from CPLEX while solving the MILP
        self.isRun: bool = False  # Boolean indicating if the solver has been run
        # Decision variables
        self.sourceFlowVars: List[float] = []  # List of the flow values assigned to each source, indexed the same as the graph.sourcesArray
        self.sinkFlowVars: List[float] = []  # List of the flow values assigned to each sink, indexed the same as the graph.sinksArray
        self.arcFlowVars: Dict[Tuple[int, int], float] = {}  # Dictionary mapping (edgeIndex, arcIndex) keys to assigned flow values
        self.arcOpenedVars: Dict[Tuple[int, int], int] = {}  # Dictionary mapping (edgeIndex, arcIndex) keys to 0/1 value of if the arc was opened in the solution

    def findSolution(self, printDetails=False) -> FlowNetworkSolution:
        """Builds the model, executes the solver, and returns the solution object with one method call"""
        self.buildModel()
        self.solveModel()
        if printDetails is True:
            self.printAllSolverData()
        solution = self.writeSolution()
        return solution

    def buildModel(self) -> None:
        """Builds the decision variables, constraints, and object function of the MILP model from the graph instance"""
        # =================== DECISION VARIABLES ===================
        # Source and sink decision variables and determines flow from/to each node - Indexed on src/sink matrix
        self.sourceFlowVars = self.model.continuous_var_list(self.graph.numSources, name="s")
        self.sinkFlowVars = self.model.continuous_var_list(self.graph.numSinks, name="t")
        # Arc decision variables - Indexed on the arc matrix and determines flow on each arc
        self.arcOpenedVars = self.model.binary_var_matrix(self.graph.numEdges, self.graph.numArcsPerEdge, name="y")
        self.arcFlowVars = self.model.continuous_var_matrix(self.graph.numEdges, self.graph.numArcsPerEdge, name="a", lb=0)

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
        self.model.add_constraint(sum(self.sinkFlowVars[i] for i in range(self.graph.numSinks)) >= self.minTargetFlow,
                                  ctname="minFlow")

        # Edge opening/capacity constraints
        for i in range(self.graph.numEdges):
            for j in range(self.graph.numArcsPerEdge):
                capacity = self.graph.possibleArcCapsArray[j]
                arcID = (self.graph.edgesArray[i][0], self.graph.edgesArray[i][1], capacity)
                ctName = "a_" + str(arcID) + "_CapAndOpen"
                self.model.add_constraint(self.arcFlowVars[(i, j)] <= self.arcOpenedVars[(i, j)] * capacity,
                                          ctname=ctName)

        # Only one arc per edge can be opened constraint
        # NOTE - Can be turned on/off as an optional param of this class
        if self.isOneArcPerEdge is True:
            for i in range(self.graph.numEdges):
                ctName = "e_" + str(i) + "_OneArcPerEdge"
                self.model.add_constraint(sum(self.arcOpenedVars[(i, j)] for j in range(self.graph.numArcsPerEdge)) <= 1,
                                          ctname=ctName)

        # Capacity constraints of sources
        if self.graph.isSourceSinkCapacitated is True:
            for i in range(self.graph.numSources):
                ctName = "s_" + str(self.graph.sourcesArray[i]) + "_Cap"
                self.model.add_constraint(self.sourceFlowVars[i] <= self.graph.sourceCapsArray[i], ctname=ctName)

        # Capacity constraints of sinks
        if self.graph.isSourceSinkCapacitated is True:
            for i in range(self.graph.numSinks):
                ctName = "t_" + str(self.graph.sinksArray[i]) + "_Cap"
                self.model.add_constraint(self.sinkFlowVars[i] <= self.graph.sinkCapsArray[i], ctname=ctName)

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
            self.model.add_constraint(self.sourceFlowVars[s] ==
                                      sum(self.arcFlowVars[(m, n)] for m in outgoingIndexes
                                          for n in range(self.graph.numArcsPerEdge)) - sum(self.arcFlowVars[(i, j)]
                                          for i in incomingIndexes for j in range(self.graph.numArcsPerEdge)), ctName)
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
            self.model.add_constraint(self.sinkFlowVars[t] ==
                                      sum(self.arcFlowVars[(m, n)] for m in incomingIndexes
                                          for n in range(self.graph.numArcsPerEdge)) - sum(self.arcFlowVars[(i, j)]
                                          for i in outgoingIndexes for j in range(self.graph.numArcsPerEdge)), ctName)
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
            self.model.add_constraint(sum(self.arcFlowVars[(i, j)] for i in incomingIndexes
                                          for j in range(self.graph.numArcsPerEdge)) - sum(self.arcFlowVars[(m, n)]
                                          for m in outgoingIndexes for n in range(self.graph.numArcsPerEdge)) == 0,
                                            ctname=ctName)

        # =================== OBJECTIVE FUNCTION ===================
        if self.graph.isSourceSinkCharged is True:
            self.model.set_objective("min", sum(self.arcFlowVars[(i, j)] * self.graph.getArcVariableCostFromEdgeCapIndices(i, j)
                                                for i in range(self.graph.numEdges)
                                                for j in range(self.graph.numArcsPerEdge)) +
                                            sum(self.arcOpenedVars[(m, n)] * self.graph.getArcFixedCostFromEdgeCapIndices(m, n)
                                                for m in range(self.graph.numEdges)
                                                for n in range(self.graph.numArcsPerEdge)) +
                                            sum(self.sourceFlowVars[s] * self.graph.sourceVariableCostsArray[s]
                                                for s in range(self.graph.numSources)) +
                                            sum(self.sinkFlowVars[t] * self.graph.sinkVariableCostsArray[t]
                                                for t in range(self.graph.numSinks)))
        elif self.graph.isSourceSinkCharged is False:
            self.model.set_objective("min", sum(self.arcFlowVars[(i, j)] * self.graph.getArcVariableCostFromEdgeCapIndices(i, j)
                                                for i in range(self.graph.numEdges)
                                                for j in range(self.graph.numArcsPerEdge)) +
                                            sum(self.arcOpenedVars[(m, n)] * self.graph.getArcFixedCostFromEdgeCapIndices(m, n)
                                                for m in range(self.graph.numEdges)
                                                for n in range(self.graph.numArcsPerEdge)))

    def setTimeLimit(self, timeLimitInSeconds: float) -> None:
        """Sets the time limit, in seconds, for the CPLEX MILP model to run"""
        print("Setting CPLEX time limit to " + str(timeLimitInSeconds) + " seconds...")
        self.model.set_time_limit(timeLimitInSeconds)

    def getTimeLimit(self) -> float:
        """Gets the time limit, in seconds, for the CPLEX MILP model to run"""
        return self.model.get_time_limit()

    def solveModel(self) -> None:
        """Solves the MILP model in CPLEX"""
        self.model.solve()
        self.isRun = True
        self.verifyTrueCost()
        self.extractRuntimeData()

    def verifyTrueCost(self) -> float:
        """Checks the true cost of the MILP's output against the solver's returned objective value"""
        srcFlows = self.model.solution.get_value_list(self.sourceFlowVars)
        sinkFlows = self.model.solution.get_value_list(self.sinkFlowVars)
        arcFlows = self.model.solution.get_value_dict(self.arcFlowVars)
        trueCost = 0.0
        if self.isSourceSinkCharged is True:
            for s in range(self.graph.numSources):
                trueCost += self.graph.sourceVariableCostsArray[s] * srcFlows[s]
            for t in range(self.graph.numSinks):
                trueCost += self.graph.sinkVariableCostsArray[t] * sinkFlows[t]
        for edge in range(self.graph.numEdges):
            for cap in range(self.graph.numArcsPerEdge):
                # Try/Except/Else block as CPLEX sometimes fails to write flow decision variables to the solution object
                try:
                    if arcFlows[(edge, cap)] > 0.0:
                        arcVariableCost = self.graph.getArcVariableCostFromEdgeCapIndices(edge, cap)
                        arcFixedCost = self.graph.getArcFixedCostFromEdgeCapIndices(edge, cap)
                        trueCost += arcVariableCost * arcFlows[(edge, cap)] + arcFixedCost
                except KeyError:
                    print("ERROR: Key error on solution.arcFlows[" + str((edge, cap)) + "]! Assuming CPLEX decided zero flow...")
        # Print verification check
        if round(self.model.solution.get_objective_value()) == round(trueCost):
            print("\nMILP cost is verified accurate!\n")
        else:
            print("\nERROR - TRUE COST OF SOLUTION DOES NOT MATCH THE SOLVER'S OBJECTIVE VALUE!")
            print("True Cost = " + str(trueCost))
            print("Solver Obj. Value = " + str(self.model.solution.get_objective_value()))
        return trueCost

    def extractRuntimeData(self) -> None:
        """Pulls runtime data out of the progress listener after solving"""
        cplexRuntimeDate = self.progressDataRecorder.recorded
        self.runtimeTimestamps = []
        for progressData in cplexRuntimeDate:
            self.runtimeTimestamps.append(progressData[8])  # Index 8 is the timestamp in seconds
            self.runtimeObjectiveValues.append(progressData[2])  # Index 2 is the current objective value
            self.runtimeBestBounds.append(progressData[3])  # Index 3 is the current known best bound
            self.runtimeGaps.append(progressData[4])  # Index 4 is the current gap

    def writeSolution(self) -> FlowNetworkSolution:
        """Writes out the solution instance"""
        if self.isRun is False:
            print("You must run the solver before building a solution!")
        elif self.model.solution is not None:
            # print("Building solution...")  # PRINT OPTION
            objValue = self.model.solution.get_objective_value()
            srcFlows = self.model.solution.get_value_list(self.sourceFlowVars)
            sinkFlows = self.model.solution.get_value_list(self.sinkFlowVars)
            arcFlows = self.model.solution.get_value_dict(self.arcFlowVars)
            thisSolution = FlowNetworkSolution(self.graph, self.minTargetFlow, objValue, objValue, srcFlows,
                                               sinkFlows, arcFlows, "cplex_milp", self.isOneArcPerEdge,
                                               self.isSourceSinkCapacitated, self.isSourceSinkCharged,
                                               optionalDescription=str(self.model.get_solve_details()))
            # print("Solution built!")  # PRINT OPTION
            return thisSolution
        else:
            print("No feasible solution exists!")

    def getObjectiveValue(self) -> float:
        """Returns the objective value found by the CPLEX MILP solver"""
        return self.model.solution.get_objective_value()

    def getCplexStatus(self) -> str:
        """Returns the status of the solution found by the CPLEX MILP solver"""
        return self.model.solve_details.status

    def getCplexStatusCode(self) -> int:
        """Returns the status code of the solution found by the CPLEX MILP solver. See the following for mapping codes:
        https://www.ibm.com/docs/en/icos/20.1.0?topic=micclcarm-solution-status-codes-by-number-in-cplex-callable-library-c-api"""
        return self.model.solve_details.status_code

    def getGap(self) -> float:
        """Returns the MILP relative gap for the solution found by the CPLEX MILP solver"""
        return self.model.solve_details.gap

    def getCplexRuntime(self) -> float:
        """Returns the runtime, in seconds, of the CPLEX MILP solver"""
        return self.model.solve_details.time

    def getBestBound(self) -> float:
        """Returns the MILP best bound for the solution found by the CPLEX MILP solver"""
        return self.model.solve_details.best_bound

    def getDeterministicTime(self) -> float:
        """Returns the deterministic time (i.e. number of ticks) of the CPLEX MILP solver"""
        return self.model.solve_details.deterministic_time

    def printAllSolverData(self) -> None:
        """Prints all the data store within the solver class"""
        self.printSolverOverview()
        self.printModel()
        self.printSolution()

    def printSolverOverview(self) -> None:
        """Prints the most important and concise details of the solver, model and solution"""
        self.model.print_information()
        if self.isRun is True:
            print(self.model.get_solve_details())
            print("Solved by= " + self.model.solution.solved_by + "\n")
            self.model.print_solution()

    def printModel(self) -> None:
        """Prints all constraints of the MILP model for the FCNF instance (FOR DEBUGGING-DON'T CALL ON LARGE INPUTS)"""
        print("=============== MILP MODEL OF FCNF INSTANCE ========================")
        self.model.print_information()
        print("=============== OBJECTIVE FUNCTION ========================")
        print(self.model.get_objective_expr())
        print("=============== CONSTRAINTS ========================")
        print(self.model.get_constraint_by_name("minFlow"))
        for s in self.graph.sourcesArray:
            print(self.model.get_constraint_by_name("s_" + str(s) + "_Conserv"))
            print(self.model.get_constraint_by_name("s_" + str(s) + "_Cap"))
        for t in self.graph.sinksArray:
            print(self.model.get_constraint_by_name("t_" + str(t) + "_Conserv"))
            print(self.model.get_constraint_by_name("t_" + str(t) + "_Cap"))
        for n in self.graph.interNodesArray:
            print(self.model.get_constraint_by_name("n_" + str(n) + "_Conserv"))
        for i in range(self.graph.numEdges):
            for j in range(self.graph.numArcsPerEdge):
                capacity = self.graph.possibleArcCapsArray[j]
                arcID = (self.graph.edgesArray[i][0], self.graph.edgesArray[i][1], capacity)
                print(self.model.get_constraint_by_name("a_" + str(arcID) + "_CapAndOpen"))

    def printSolution(self) -> None:
        """Prints the solution data of the FCNF instance solved by the MILP model"""
        print("=============== SOLUTION DETAILS ========================")
        print(self.model.get_solve_details())
        if self.isRun is True:
            print("Solved by= " + self.model.solution.solved_by + "\n")
            print("=============== SOLUTION VALUES ========================")
            self.model.print_solution()
        else:
            print("No feasible solution exists!")
