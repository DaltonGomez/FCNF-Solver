from docplex.mp.model import Model


class ExactSolver:
    """Class that solves a FixedChargeFlowNetwork instance w/ parallel edges exactly via a MILP model within CPLEX 20.1"""

    def __init__(self, FCFNinstance, minTargetFlow: int):
        """Constructor of a MILP-Solver instance
        NOTE: FCFNinstance must be of type FixedChargeFlowNetwork (Not type hinted to prevent circular import)"""
        # Input attributes
        self.FCFN = FCFNinstance
        self.minTargetFlow = minTargetFlow
        # Solver model
        self.model = Model(name="FCFN-MILP-ExactSolver", log_output=False, cts_by_name=True)
        self.isRun = False

    def buildModel(self):
        """Builds the decision variables, constraints, and object function of the MILP model from the FCFN instance"""
        # =================== DECISION VARIABLES ===================
        # Source and sink decision variables
        self.model.sourceFlowVars = self.model.continuous_var_list(self.FCFN.numSources, name="s")
        self.model.sinkFlowVars = self.model.continuous_var_list(self.FCFN.numSinks, name="t")
        # Edge decision variables - Built as a matrix indexed on [edge][capacity] and with a lower bound of 0
        self.model.edgeOpenedVars = self.model.binary_var_matrix(self.FCFN.numEdges, self.FCFN.numEdgeCaps,
                                                                 name="y")
        self.model.edgeFlowVars = self.model.continuous_var_matrix(self.FCFN.numEdges, self.FCFN.numEdgeCaps,
                                                                   name="e", lb=0)

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
        self.model.add_constraint(
            sum(self.model.sinkFlowVars[i] for i in range(self.FCFN.numSinks)) >= self.minTargetFlow,
            ctname="minFlow")

        # Edge opening/capacity constraints
        for i in range(self.FCFN.numEdges):
            for j in range(self.FCFN.numEdgeCaps):
                ctName = "e" + str(i) + "_" + str(j) + "CapAndOpen"
                edgeCapacity = self.FCFN.edgeCaps[j]
                self.model.add_constraint(
                    self.model.edgeFlowVars[(i, j)] <= self.model.edgeOpenedVars[(i, j)] * edgeCapacity,
                    ctname=ctName)

        # Capacity constraints of sources
        for i in range(self.FCFN.numSources):
            ctName = "s" + str(i) + "Cap"
            srcCapacity = self.FCFN.nodesDict["s" + str(i)].capacity
            self.model.add_constraint(self.model.sourceFlowVars[i] <= srcCapacity, ctname=ctName)

        # Capacity constraints of sinks
        for i in range(self.FCFN.numSinks):
            ctName = "t" + str(i) + "Cap"
            sinkCapacity = self.FCFN.nodesDict["t" + str(i)].capacity
            self.model.add_constraint(self.model.sinkFlowVars[i] <= sinkCapacity, ctname=ctName)

        # Only one capacity per edge constraints
        # TODO - Remove/comment out this constraint as it can be implied by convex capacity vs. cost functions?
        for i in range(self.FCFN.numEdges):
            ctName = "e" + str(i) + "CapPerEdge"
            self.model.add_constraint(
                sum(self.model.edgeOpenedVars[(i, j)] for j in range(self.FCFN.numEdgeCaps)) <= 1,
                ctname=ctName)

        # Conservation of flow constraints
        for node in self.FCFN.nodesDict:
            nodeObj = self.FCFN.nodesDict[node]
            nodeType = node.strip("0123456789")
            nodeID = node.strip("stn")
            # Get outgoing and incoming edge number ids
            outgoingIDs = []
            for outgoingEdge in nodeObj.outgoingEdges:
                outgoingIDs.append(int(outgoingEdge.strip("e")))
            incomingIDs = []
            for incomingEdge in nodeObj.incomingEdges:
                incomingIDs.append(int(incomingEdge.strip("e")))
            # Source flow conservation
            if nodeType == "s":
                ctName = "s" + str(nodeID) + "Conserv"
                self.model.add_constraint(self.model.sourceFlowVars[int(nodeID)] ==
                                          sum(self.model.edgeFlowVars[(i, j)] for i in outgoingIDs for j in
                                              range(self.FCFN.numEdgeCaps)),
                                          ctname=ctName)
            # Sink flow conservation
            elif nodeType == "t":
                ctName = "t" + str(nodeID) + "Conserv"
                self.model.add_constraint(self.model.sinkFlowVars[int(nodeID)] ==
                                          sum(self.model.edgeFlowVars[(i, j)] for i in incomingIDs for j in
                                              range(self.FCFN.numEdgeCaps)),
                                          ctname=ctName)
            # Transshipment flow conservation
            elif nodeType == "n":
                ctName = "n" + str(nodeID) + "Conserv"
                self.model.add_constraint(sum(
                    self.model.edgeFlowVars[(i, j)] for i in incomingIDs for j in range(self.FCFN.numEdgeCaps)) - sum(
                    self.model.edgeFlowVars[(m, n)] for m in outgoingIDs for n in range(self.FCFN.numEdgeCaps)) == 0,
                                          ctname=ctName)
        # =================== OBJECTIVE FUNCTION ===================
        self.model.set_objective("min",
                                 sum(self.model.sourceFlowVars[i] * self.FCFN.nodesDict["s" + str(i)].variableCost for
                                     i
                                     in range(self.FCFN.numSources))
                                 + sum(
                                     self.model.sinkFlowVars[j] * self.FCFN.nodesDict["t" + str(j)].variableCost for j
                                     in range(self.FCFN.numSinks)) + sum(
                                     self.model.edgeFlowVars[(m, n)] * self.FCFN.edgeVariableCosts[n] for m
                                     in range(self.FCFN.numEdges) for n in range(self.FCFN.numEdgeCaps)) + sum(
                                     self.model.edgeOpenedVars[(a, b)] * self.FCFN.edgeFixedCosts[b] for a in
                                     range(self.FCFN.numEdges) for b in range(self.FCFN.numEdgeCaps)))

    def solveModel(self):
        """Solves the MILP model in CPLEX"""
        print("Attempting to solve model...")
        self.model.solve()
        self.isRun = True
        print("Solver execution complete...\n")

    def writeSolution(self):
        """Writes the solution to the FCFN instance by updating output attributes across the FCFN, nodes, and edges"""
        if self.model.solution is not None:
            # Disperse solution results back to FCFN
            self.FCFN.isSolved = True
            self.FCFN.minTargetFlow = self.minTargetFlow
            self.FCFN.totalCost = self.model.solution.get_objective_value()
            self.FCFN.totalFlow = sum(self.model.solution.get_value_list(self.model.sinkFlowVars))
            # Disperse solution results back to sources
            sourceValues = self.model.solution.get_value_list(self.model.sourceFlowVars)
            for i in range(self.FCFN.numSources):
                thisSource = self.FCFN.nodesDict["s" + str(i)]
                if sourceValues[i] > 0:
                    thisSource.opened = True
                    thisSource.flow = sourceValues[i]
                    thisSource.totalCost = thisSource.flow * thisSource.variableCost
            # Disperse solution results back to sources
            sinkValues = self.model.solution.get_value_list(self.model.sinkFlowVars)
            for i in range(self.FCFN.numSinks):
                thisSink = self.FCFN.nodesDict["t" + str(i)]
                if sinkValues[i] > 0:
                    thisSink.opened = True
                    thisSink.flow = sinkValues[i]
                    thisSink.totalCost = thisSink.flow * thisSink.variableCost
            # Disperse solution results back to edges
            edgeValues = self.model.solution.get_value_dict(self.model.edgeFlowVars)
            for i in range(self.FCFN.numEdges):
                thisEdge = self.FCFN.edgesDict["e" + str(i)]
                for j in range(self.FCFN.numEdgeCaps):
                    if edgeValues[(i, j)] > 0:
                        thisEdge.opened = True
                        thisEdge.capacity = self.FCFN.edgeCaps[j]
                        thisEdge.fixedCost = self.FCFN.edgeFixedCosts[j]
                        thisEdge.variableCost = self.FCFN.edgeVariableCosts[j]
                        thisEdge.flow = edgeValues[(i, j)]
                        thisEdge.totalCost = thisEdge.flow * thisEdge.variableCost + thisEdge.fixedCost
            # Disperse solution results back to intermediate nodes
            for i in range(self.FCFN.numIntermediateNodes):
                thisNode = self.FCFN.nodesDict["n" + str(i)]
                for edge in thisNode.incomingEdges:
                    thisNode.flow += self.FCFN.edgesDict[edge].flow
                if thisNode.flow > 0:
                    thisNode.opened = True
        else:
            print("No feasible solution exists!")

    def printSolverOverview(self):
        """Prints the most important and concise details of the solver, model and solution"""
        self.model.print_information()
        if self.isRun is True:
            print("Total Cost: " + str(self.FCFN.totalCost))
            print("Total Flow: " + str(self.FCFN.totalFlow))

    def printModel(self):
        """Prints all constraints of the MILP model for the FCFN instance (FOR DEBUGGING- DON'T CALL ON LARGE INPUTS)"""
        print("=============== MILP MODEL OF FCNF INSTANCE ========================")
        self.model.print_information()
        print("=============== OBJECTIVE FUNCTION ========================")
        print(self.model.get_objective_expr())
        print("=============== CONSTRAINTS ========================")
        print(self.model.get_constraint_by_name("minFlow"))
        for i in range(self.FCFN.numSources):
            print(self.model.get_constraint_by_name("s" + str(i) + "Conserv"))
            print(self.model.get_constraint_by_name("s" + str(i) + "Cap"))
        for i in range(self.FCFN.numSinks):
            print(self.model.get_constraint_by_name("t" + str(i) + "Conserv"))
            print(self.model.get_constraint_by_name("t" + str(i) + "Cap"))
        for i in range(self.FCFN.numSinks):
            print(self.model.get_constraint_by_name("t" + str(i) + "Conserv"))
            print(self.model.get_constraint_by_name("t" + str(i) + "Cap"))
        for i in range(self.FCFN.numNodes - (self.FCFN.numSources + self.FCFN.numSinks)):
            print(self.model.get_constraint_by_name("n" + str(i) + "Conserv"))
        for i in range(self.FCFN.numEdges):
            print(self.model.get_constraint_by_name("e" + str(i) + "CapPerEdge"))
            for j in range(self.FCFN.numEdgeCaps):
                print(self.model.get_constraint_by_name("e" + str(i) + "_" + str(j) + "CapAndOpen"))

    def printSolution(self):
        """Prints the solution data of the FCFN instance solved by the MILP model"""
        print("=============== SOLUTION DETAILS ========================")
        print(self.model.get_solve_details())
        if self.FCFN.isSolved is True:
            print("Solved by= " + self.model.solution.solved_by + "\n")
            print("=============== SOLUTION VALUES ========================")
            self.model.print_solution()
            print("Total Cost: " + str(self.FCFN.totalCost))
            print("Total Flow: " + str(self.FCFN.totalFlow))
        else:
            print("No feasible solution exists!")
