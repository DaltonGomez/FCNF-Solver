from docplex.mp.model import Model

from src.Network.FlowNetwork import FlowNetwork
from src.Network.Solution import Solution


class MILPsolverCPLEX:
    """Class that solves a FCFN instance exactly via a MILP model within CPLEX"""

    def __init__(self, network: FlowNetwork, minTargetFlow: int, isOneArcPerEdge=True, isSrcSinkConstrained=True,
                 isSrcSinkCharged=True):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.network = network
        self.minTargetFlow = minTargetFlow
        self.isOneArcPerEdge = isOneArcPerEdge
        self.isSrcSinkConstrained = isSrcSinkConstrained
        self.isSrcSinkCharged = isSrcSinkCharged
        # Solver model
        self.model = Model(name="FCFN-MILP-ExactSolver", log_output=False, cts_by_name=True)
        self.isRun = False
        # Decision variables
        self.sourceFlowVars = None
        self.sinkFlowVars = None
        self.arcOpenedVars = None
        self.arcFlowVars = None

    def buildModel(self) -> None:
        """Builds the decision variables, constraints, and object function of the MILP model from the FCFN instance"""
        # =================== DECISION VARIABLES ===================
        # Source and sink decision variables and determines flow from/to each node - Indexed on src/sink matrix
        self.sourceFlowVars = self.model.continuous_var_list(self.network.numSources, name="s")
        self.sinkFlowVars = self.model.continuous_var_list(self.network.numSinks, name="t")
        # Arc decision variables - Indexed on the arc matrix and determines flow on each arc
        self.arcOpenedVars = self.model.binary_var_matrix(self.network.numEdges, self.network.numArcCaps, name="y")
        self.arcFlowVars = self.model.continuous_var_matrix(self.network.numEdges, self.network.numArcCaps, name="a",
                                                            lb=0)

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
        self.model.add_constraint(sum(self.sinkFlowVars[i] for i in range(self.network.numSinks)) >= self.minTargetFlow,
                                  ctname="minFlow")

        # Edge opening/capacity constraints
        for i in range(self.network.numEdges):
            for j in range(self.network.numArcCaps):
                capacity = self.network.possibleArcCapsArray[j]
                arcID = (self.network.edgesArray[i][0], self.network.edgesArray[i][1], capacity)
                ctName = "a_" + str(arcID) + "_CapAndOpen"
                self.model.add_constraint(self.arcFlowVars[(i, j)] <= self.arcOpenedVars[(i, j)] * capacity,
                                          ctname=ctName)

        # Only one arc per edge can be opened constraint (NOTE: Can be turned on/off as an optional param of this class)
        if self.isOneArcPerEdge is True:
            for i in range(self.network.numEdges):
                ctName = "e_" + str(i) + "_OneArcPerEdge"
                self.model.add_constraint(sum(self.arcOpenedVars[(i, j)] for j in range(self.network.numArcCaps)) <= 1,
                                          ctname=ctName)

        # Capacity constraints of sources
        if self.network.isSourceSinkCapacitated is True:
            for i in range(self.network.numSources):
                ctName = "s_" + str(self.network.sourcesArray[i]) + "_Cap"
                self.model.add_constraint(self.sourceFlowVars[i] <= self.network.sourceCapsArray[i], ctname=ctName)

        # Capacity constraints of sinks
        if self.network.isSourceSinkCapacitated is True:
            for i in range(self.network.numSinks):
                ctName = "t_" + str(self.network.sinksArray[i]) + "_Cap"
                self.model.add_constraint(self.sinkFlowVars[i] <= self.network.sinkCapsArray[i], ctname=ctName)

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
            ctName = "s_" + str(source) + "_Conserv"
            self.model.add_constraint(self.sourceFlowVars[s] ==
                                      sum(self.arcFlowVars[(m, n)] for m in outgoingIndexes
                                          for n in range(self.network.numArcCaps)) - sum(self.arcFlowVars[(i, j)]
                                                                                         for i in incomingIndexes
                                                                                         for j in range(
                self.network.numArcCaps)), ctName)
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
            ctName = "t_" + str(sink) + "_Conserv"
            self.model.add_constraint(self.sinkFlowVars[t] ==
                                      sum(self.arcFlowVars[(m, n)] for m in incomingIndexes
                                          for n in range(self.network.numArcCaps)) - sum(self.arcFlowVars[(i, j)]
                                                                                         for i in outgoingIndexes
                                                                                         for j in range(
                self.network.numArcCaps)), ctName)
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
            ctName = "n_" + str(interNode) + "_Conserv"
            self.model.add_constraint(sum(self.arcFlowVars[(i, j)] for i in incomingIndexes
                                          for j in range(self.network.numArcCaps)) - sum(
                self.arcFlowVars[(m, n)] for m in outgoingIndexes
                for n in range(self.network.numArcCaps)) == 0,
                                      ctname=ctName)

        # =================== OBJECTIVE FUNCTION ===================
        # NOTE: Borderline unreadable but verified correct
        if self.network.isSourceSinkCharged is True:
            self.model.set_objective("min", sum(self.arcFlowVars[(i, j)]
                                                * self.network.arcsMatrix[self.network.arcsDict[(
                self.network.edgesArray[i][0], self.network.edgesArray[i][1],
                self.network.possibleArcCapsArray[j])].numID][6] for i in range(self.network.numEdges) for j in
                                                range(self.network.numArcCaps)) + sum(
                self.arcOpenedVars[(m, n)] *
                self.network.arcsMatrix[self.network.arcsDict[(self.network.edgesArray[m][0],
                                                               self.network.edgesArray[m][1],
                                                               self.network.possibleArcCapsArray[
                                                                   n])].numID][5] for m in
                range(self.network.numEdges) for n in
                range(self.network.numArcCaps)) + sum(
                self.sourceFlowVars[s] * self.network.sourceVariableCostsArray[s] for s in
                range(self.network.numSources)) +
                                     sum(self.sinkFlowVars[t] * self.network.sinkVariableCostsArray[t] for t in
                                         range(self.network.numSinks)))
        elif self.network.isSourceSinkCharged is False:
            self.model.set_objective("min",
                                     sum(self.arcFlowVars[(i, j)] * self.network.arcsMatrix[self.network.arcsDict[(
                                         self.network.edgesArray[i][0], self.network.edgesArray[i][1],
                                         self.network.possibleArcCapsArray[j])].numID][6] for i in
                                         range(self.network.numEdges) for j in
                                         range(self.network.numArcCaps)) + sum(self.arcOpenedVars[(m, n)] *
                                                                               self.network.arcsMatrix[
                                                                                   self.network.arcsDict[
                                                                                       (self.network.edgesArray[m][0],
                                                                                        self.network.edgesArray[m][1],
                                                                                        self.network.possibleArcCapsArray[
                                                                                            n])].numID][5] for m in
                                                                               range(self.network.numEdges) for n
                                                                               in range(self.network.numArcCaps)))

    def solveModel(self) -> None:
        """Solves the MILP model in CPLEX"""
        print("\nAttempting to solve model...")
        self.model.solve()
        self.isRun = True
        print("Solver execution complete...\n")

    def writeSolution(self) -> Solution:
        """Saves the solution instance """
        if self.isRun is False:
            print("You must run the solver before building a solution!")
        elif self.model.solution is not None:
            print("Building solution...")
            objValue = self.model.solution.get_objective_value()
            srcFlows = self.model.solution.get_value_list(self.sourceFlowVars)
            sinkFlows = self.model.solution.get_value_list(self.sinkFlowVars)
            arcFlows = self.model.solution.get_value_dict(self.arcFlowVars)
            arcsOpen = self.model.solution.get_value_dict(self.arcOpenedVars)
            thisSolution = Solution(self.network, self.minTargetFlow, objValue, srcFlows, sinkFlows, arcFlows,
                                    arcsOpen, "cplex_milp", self.isOneArcPerEdge, self.isSrcSinkConstrained,
                                    self.isSrcSinkCharged, optionalDescription=str(self.model.get_solve_details()))
            return thisSolution
        else:
            print("No feasible solution exists!")

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
        """Prints all constraints of the MILP model for the FCFN instance (FOR DEBUGGING- DON'T CALL ON LARGE INPUTS)"""
        print("=============== MILP MODEL OF FCNF INSTANCE ========================")
        self.model.print_information()
        print("=============== OBJECTIVE FUNCTION ========================")
        print(self.model.get_objective_expr())
        print("=============== CONSTRAINTS ========================")
        print(self.model.get_constraint_by_name("minFlow"))
        for s in self.network.sourcesArray:
            print(self.model.get_constraint_by_name("s_" + str(s) + "_Conserv"))
            print(self.model.get_constraint_by_name("s_" + str(s) + "_Cap"))
        for t in self.network.sinksArray:
            print(self.model.get_constraint_by_name("t_" + str(t) + "_Conserv"))
            print(self.model.get_constraint_by_name("t_" + str(t) + "_Cap"))
        for n in self.network.interNodesArray:
            print(self.model.get_constraint_by_name("n_" + str(n) + "_Conserv"))
        for i in range(self.network.numEdges):
            for j in range(self.network.numArcCaps):
                capacity = self.network.possibleArcCapsArray[j]
                arcID = (self.network.edgesArray[i][0], self.network.edgesArray[i][1], capacity)
                print(self.model.get_constraint_by_name("a_" + str(arcID) + "_CapAndOpen"))

    def printSolution(self) -> None:
        """Prints the solution data of the FCFN instance solved by the MILP model"""
        print("=============== SOLUTION DETAILS ========================")
        print(self.model.get_solve_details())
        if self.isRun is True:
            print("Solved by= " + self.model.solution.solved_by + "\n")
            print("=============== SOLUTION VALUES ========================")
            self.model.print_solution()
        else:
            print("No feasible solution exists!")
