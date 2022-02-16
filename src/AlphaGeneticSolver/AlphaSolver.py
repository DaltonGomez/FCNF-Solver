from docplex.mp.model import Model


class AlphaSolver:
    """Class that solves an alpha-relaxed FCFN instance w/ via a LP model within CPLEX"""

    def __init__(self, alphaIndividual, minTargetFlow: int):
        """Constructor of an AlphaSolver instance
        NOTE: alphaIndividual must be of type AlphaIndividual (Not type hinted to prevent circular import)"""
        # Input attributes
        self.individual = alphaIndividual
        self.minTargetFlow = minTargetFlow
        # Solver model
        self.model = Model(name="AlphaFCFN-LP-RelaxedSolver", log_output=False, cts_by_name=True)
        self.isRun = False

    def buildModel(self) -> None:
        """Builds the decision variables, constraints, and object function of the LP model from the AlphaIndividual instance"""
        # =================== DECISION VARIABLES ===================
        # Source and sink decision variables
        self.model.sourceFlowVars = self.model.continuous_var_list(self.individual.FCFN.numSources, name="s")
        self.model.sinkFlowVars = self.model.continuous_var_list(self.individual.FCFN.numSinks, name="t")
        self.model.edgeFlowVars = self.model.continuous_var_list(self.individual.FCFN.numEdges, name="e", lb=0)

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
        self.model.add_constraint(
            sum(self.model.sinkFlowVars[i] for i in range(self.individual.FCFN.numSinks)) >= self.minTargetFlow,
            ctname="minFlow")

        # Edge opening/capacity constraints
        for i in range(self.individual.FCFN.numEdges):
            ctName = "e" + str(i) + "Cap"
            edgeCapacity = self.individual.FCFN.edgesDict["e" + str(i)].capacity
            self.model.add_constraint(self.model.edgeFlowVars[i] <= edgeCapacity, ctname=ctName)

        # Capacity constraints of sources
        for i in range(self.individual.FCFN.numSources):
            ctName = "s" + str(i) + "Cap"
            srcCapacity = self.individual.FCFN.nodesDict["s" + str(i)].capacity
            self.model.add_constraint(self.model.sourceFlowVars[i] <= srcCapacity, ctname=ctName)

        # Capacity constraints of sinks
        for i in range(self.individual.FCFN.numSinks):
            ctName = "t" + str(i) + "Cap"
            sinkCapacity = self.individual.FCFN.nodesDict["t" + str(i)].capacity
            self.model.add_constraint(self.model.sinkFlowVars[i] <= sinkCapacity, ctname=ctName)

        # Conservation of flow constraints
        for node in self.individual.FCFN.nodesDict:
            nodeObj = self.individual.FCFN.nodesDict[node]
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
                                          sum(self.model.edgeFlowVars[i] for i in outgoingIDs),
                                          ctname=ctName)
            # Sink flow conservation
            elif nodeType == "t":
                ctName = "t" + str(nodeID) + "Conserv"
                self.model.add_constraint(self.model.sinkFlowVars[int(nodeID)] ==
                                          sum(self.model.edgeFlowVars[i] for i in incomingIDs),
                                          ctname=ctName)
            # Transshipment flow conservation
            elif nodeType == "n":
                ctName = "n" + str(nodeID) + "Conserv"
                self.model.add_constraint(sum(
                    self.model.edgeFlowVars[i] for i in incomingIDs) - sum(
                    self.model.edgeFlowVars[j] for j in outgoingIDs) == 0,
                                          ctname=ctName)
        # =================== OBJECTIVE FUNCTION ===================
        self.model.set_objective("min",
                                 sum(self.model.sourceFlowVars[i] * self.individual.FCFN.nodesDict[
                                     "s" + str(i)].variableCost for
                                     i
                                     in range(self.individual.FCFN.numSources))
                                 + sum(
                                     self.model.sinkFlowVars[j] * self.individual.FCFN.nodesDict[
                                         "t" + str(j)].variableCost for j
                                     in range(self.individual.FCFN.numSinks)) + sum(
                                     self.model.edgeFlowVars[n] * (
                                             self.individual.FCFN.edgesDict["e" + str(n)].variableCost +
                                             self.individual.alphaValues[n] *
                                             self.individual.FCFN.edgesDict["e" + str(n)].fixedCost) for n in
                                     range(self.individual.FCFN.numEdges)))

    def solveModel(self) -> None:
        """Solves the alpha-relaxed LP model in CPLEX"""
        # print("Attempting to solve model...")
        self.model.solve()
        self.isRun = True
        # print("Solver execution complete...\n")

    def writeSolution(self) -> None:
        """Writes the solution to the individual instance by updating output attributes across the FCFN, nodes, and edges"""
        if self.model.solution is not None:
            # Disperse solution results back to individual
            self.individual.isSolved = True
            self.individual.minTargetFlow = self.minTargetFlow
            self.individual.fakeCost = self.model.solution.get_objective_value()
            self.individual.totalFlow = sum(self.model.solution.get_value_list(self.model.sinkFlowVars))
            # Disperse solution results back to sources
            sourceValues = self.model.solution.get_value_list(self.model.sourceFlowVars)
            for i in range(self.individual.FCFN.numSources):
                thisSource = self.individual.FCFN.nodesDict["s" + str(i)]
                if sourceValues[i] > 0:
                    thisSource.opened = True
                    thisSource.flow = sourceValues[i]
                    thisSource.totalCost = thisSource.flow * thisSource.variableCost
            # Disperse solution results back to sources
            sinkValues = self.model.solution.get_value_list(self.model.sinkFlowVars)
            for i in range(self.individual.FCFN.numSinks):
                thisSink = self.individual.FCFN.nodesDict["t" + str(i)]
                if sinkValues[i] > 0:
                    thisSink.opened = True
                    thisSink.flow = sinkValues[i]
                    thisSink.totalCost = thisSink.flow * thisSink.variableCost
            # Disperse solution results back to edges
            edgeValues = self.model.solution.get_value_list(self.model.edgeFlowVars)
            for i in range(self.individual.FCFN.numEdges):
                thisEdge = self.individual.FCFN.edgesDict["e" + str(i)]
                if edgeValues[i] > 0:
                    thisEdge.opened = True
                    thisEdge.flow = edgeValues[i]
                    thisEdge.totalCost = thisEdge.flow * thisEdge.variableCost + thisEdge.fixedCost
            # Disperse solution results back to intermediate nodes
            for i in range(self.individual.FCFN.numIntermediateNodes):
                thisNode = self.individual.FCFN.nodesDict["n" + str(i)]
                for edge in thisNode.incomingEdges:
                    thisNode.flow += self.individual.FCFN.edgesDict[edge].flow
                if thisNode.flow > 0:
                    thisNode.opened = True
            # Compute the true cost of the solution under the FCFN model
            self.individual.calculateTrueCost()
        else:
            print("No feasible solution exists!")

    def printSolverOverview(self) -> None:
        """Prints the most important and concise details of the solver, model and solution"""
        self.model.print_information()
        if self.isRun is True:
            self.model.print_solution()
            print("Fake Cost: " + str(self.individual.fakeCost))
            print("Total Flow: " + str(self.individual.totalFlow))
            print("True Cost: " + str(self.individual.trueCost))

    def printModel(self) -> None:
        """Prints all constraints of the alpha-relaxed LP model for the Individual instance (FOR DEBUGGING- DON'T CALL ON LARGE INPUTS)"""
        print("=============== MILP MODEL OF FCNF INSTANCE ========================")
        self.model.print_information()
        print("=============== OBJECTIVE FUNCTION ========================")
        print(self.model.get_objective_expr())
        print("=============== CONSTRAINTS ========================")
        print(self.model.get_constraint_by_name("minFlow"))
        for i in range(self.individual.FCFN.numSources):
            print(self.model.get_constraint_by_name("s" + str(i) + "Conserv"))
            print(self.model.get_constraint_by_name("s" + str(i) + "Cap"))
        for i in range(self.individual.FCFN.numSinks):
            print(self.model.get_constraint_by_name("t" + str(i) + "Conserv"))
            print(self.model.get_constraint_by_name("t" + str(i) + "Cap"))
        for i in range(self.individual.FCFN.numSinks):
            print(self.model.get_constraint_by_name("t" + str(i) + "Conserv"))
            print(self.model.get_constraint_by_name("t" + str(i) + "Cap"))
        for i in range(self.individual.FCFN.numIntermediateNodes):
            print(self.model.get_constraint_by_name("n" + str(i) + "Conserv"))
        for i in range(self.individual.FCFN.numEdges):
            print(self.model.get_constraint_by_name("e" + str(i) + "Cap"))

    def printSolution(self) -> None:
        """Prints the solution data of the Individual instance solved by the alpha-relaxed LP model"""
        print("=============== SOLUTION DETAILS ========================")
        print(self.model.get_solve_details())
        if self.individual.FCFN.isSolved is True:
            print("Solved by= " + self.model.solution.solved_by + "\n")
            print("=============== SOLUTION VALUES ========================")
            self.model.print_solution()
            print("Total Cost: " + str(self.individual.FCFN.totalCost))
            print("Total Flow: " + str(self.individual.FCFN.totalFlow))
        else:
            print("No feasible solution exists!")
