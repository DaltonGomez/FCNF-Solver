from typing import List, Dict, Tuple

from docplex.mp.model import Model
from docplex.mp.progress import ProgressDataRecorder

from TransportationReduction.TransportationProblem import TransportationProblem


class FCTPsolverCPLEX:
    """Class that solves a fixed-charge transportation problem reduction optimally via a MILP model within CPLEX"""

    def __init__(self, transportProblem: TransportationProblem, minTargetFlow: float, logOutput=False):
        """Constructor of a FCTPsolverCPLEX instance"""
        # Input attributes
        self.transportProblem: TransportationProblem = transportProblem
        self.minTargetFlow: float = minTargetFlow
        # Solver model
        self.model: Model = Model(name="FCTP-MILP-Solver", log_output=logOutput, cts_by_name=True)
        self.progressDataRecorder: ProgressDataRecorder = ProgressDataRecorder(clock="gap")
        self.model.add_progress_listener(self.progressDataRecorder)
        self.runtimeTimestamps: List = []
        self.runtimeObjectiveValues: List = []
        self.runtimeBestBounds: List = []
        self.runtimeGaps: List = []
        self.isRun: bool = False
        # Dictionary mapping transport arc IDs as ((fromNode, toNode, cap) transportDestination) keys to assigned flow values
        self.transportArcOpenedVars: Dict[Tuple[Tuple[int, int, float], int], int] = {}
        self.transportArcFlowVars: Dict[Tuple[Tuple[int, int, float], int], float] = {}

    def findSolution(self, printDetails=False) -> None:
        """Builds the model, executes the solver, and returns the solution object with one method call"""
        # TODO - Revise to include solution writing
        self.buildModel()
        self.solveModel()
        if printDetails is True:
            self.printAllSolverData()

    def buildModel(self) -> None:
        """Builds the decision variables, constraints, and object function of the MILP model from the graph instance"""
        # =================== DECISION VARIABLES ===================
        self.transportArcOpenedVars = self.model.binary_var_dict(self.transportProblem.transportArcs.keys(), name="y")
        self.transportArcFlowVars = self.model.continuous_var_dict(self.transportProblem.transportArcs.keys(), name="q", lb=0)

        # =================== CONSTRAINTS ===================
        # Origin supply satisfaction constraint
        for originID in self.transportProblem.origins.keys():
            originCtname = str(originID) + "_originSupply"
            self.model.add_constraint(self.transportArcFlowVars[(originID, originID[0])] + self.transportArcFlowVars[(originID, originID[1])], ctname=originCtname)

        # Destination demand satisfaction constraint
        for destinationID in self.transportProblem.destinations.keys():
            destCtname = str(destinationID) + "_destDemand"
            destinationObj = self.transportProblem.destinations[destinationID]
            self.model.add_constraint(sum(self.transportArcFlowVars[incomingTransportArc] for incomingTransportArc in destinationObj.incomingTransportArcs), ctname=destCtname)

        # Edge opening/capacity constraints
        for transportArcID in self.transportProblem.transportArcs.keys():
            arcCtname = str(transportArcID) + "_transportArc"
            originSupply = self.transportProblem.origins[transportArcID[0]].supply
            destinationDemand = self.transportProblem.destinations[transportArcID[1]].demand
            transportArcCap = min(originSupply, destinationDemand)
            self.model.add_constraint(self.transportArcFlowVars[transportArcID] <= self.transportArcOpenedVars[transportArcID] * transportArcCap, ctname=arcCtname)

        # =================== OBJECTIVE FUNCTION ===================
        self.model.set_objective("min",
            sum(self.transportArcFlowVars[transportArcID] * self.transportProblem.transportArcs[transportArcID].variableCost
                for transportArcID in self.transportProblem.transportArcs.keys()) +
            sum(self.transportArcOpenedVars[transportArcID] * self.transportProblem.transportArcs[transportArcID].fixedCost
                for transportArcID in self.transportProblem.transportArcs.keys()))

    def setTimeLimit(self, timeLimitInSeconds: float) -> None:
        """Sets the time limit, in seconds, for the CPLEX model to run"""
        print("Setting CPLEX time limit to " + str(timeLimitInSeconds) + " seconds...")
        self.model.set_time_limit(timeLimitInSeconds)

    def getTimeLimit(self) -> float:
        """Gets the time limit, in seconds, for the CPLEX model to run"""
        return self.model.get_time_limit()

    def solveModel(self) -> None:
        """Solves the MILP model in CPLEX"""
        self.model.solve()
        self.isRun = True
        # self.verifyTrueCost()
        self.extractRuntimeData()

    def extractRuntimeData(self) -> None:
        """Pulls runtime data out of the progress listener after solving"""
        cplexRuntimeDate = self.progressDataRecorder.recorded
        self.runtimeTimestamps = []
        for progressData in cplexRuntimeDate:
            self.runtimeTimestamps.append(progressData[8])  # Index 8 is the timestamp in seconds
            self.runtimeObjectiveValues.append(progressData[2])  # Index 2 is the current objective value
            self.runtimeBestBounds.append(progressData[3])  # Index 3 is the current known best bound
            self.runtimeGaps.append(progressData[4])  # Index 4 is the current gap

    def cleanArcFlowsKeyErrors(self) -> dict:
        """Iterates over CPLEX's dictionary of arc flows and resolves any key errors by assuming 0.0"""
        print("\nResolving any key errors from CPLEX before writing solution...")
        cplexArcFlows = self.model.solution.get_value_dict(self.transportArcFlowVars)
        for transportArcID in self.transportProblem.transportArcs.keys():
            # Try/Except/Else block as CPLEX sometimes fails to write flow decision variables to the arc flows dict
            try:
                if cplexArcFlows[transportArcID] >= 0.0:
                    continue
            except KeyError:
                print("ERROR: Key error on solution.arcFlows[" + str(transportArcID) + "]! Assuming CPLEX decided zero flow...")
                cplexArcFlows[transportArcID] = 0.0
        return cplexArcFlows

    def writeSolution(self) -> None:
        """Writes out the solution instance"""
        # TODO - Revise to account for solution writing
        pass
        """
        if self.isRun is False:
            print("You must run the solver before building a solution!")
        elif self.model.solution is not None:
            objValue = self.model.solution.get_objective_value()
            srcFlows = self.model.solution.get_value_list(self.sourceFlowVars)
            sinkFlows = self.model.solution.get_value_list(self.sinkFlowVars)
            arcFlows = self.cleanArcFlowsKeyErrors()
            thisSolution = FlowNetworkSolution(self.graph, self.minTargetFlow, objValue, objValue, srcFlows,
                                               sinkFlows, arcFlows, "cplex_milp", self.isOneArcPerEdge,
                                               self.isSourceSinkCapacitated, self.isSourceSinkCharged,
                                               optionalDescription=str(self.model.get_solve_details()))
            return thisSolution
        else:
            print("No feasible solution exists!")
        """

    def clearSolution(self) -> None:
        """Clears all solution data for the solver while leaving the objective and constraints unchanged"""
        self.model.solution.clear()

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
        for originID in self.transportProblem.origins.keys():
            print(self.model.get_constraint_by_name(str(originID) + "_originSupply"))
        for destID in self.transportProblem.destinations.keys():
            print(self.model.get_constraint_by_name(str(destID) + "_destDemand"))
        for transportArcID in self.transportProblem.transportArcs.keys():
            print(self.model.get_constraint_by_name(str(transportArcID) + "_transportArc"))

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
