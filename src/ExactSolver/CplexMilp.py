from docplex.mp.model import Model

from src.Network.FlowNetwork import FlowNetwork


class ExactSolver:
    """Class that solves a FCFN instance exactly via a MILP model within CPLEX"""

    def __init__(self, network: FlowNetwork, minTargetFlow: int):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.network = network
        self.minTargetFlow = minTargetFlow
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
        self.sourceFlowVars = self.model.continuous_var_dict(self.network.numSources, name="s")
        self.sinkFlowVars = self.model.continuous_var_dict(self.network.numSinks, name="t")
        # Arc decision variables - Indexed on the arc matrix and determines flow on each arc
        self.arcOpenedVars = self.model.binary_var_dict(self.network.numArcs, name="y")
        self.arcFlowVars = self.model.continuous_var_dict(self.network.numArcs, name="a", lb=0)

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
        self.model.add_constraint(sum(self.sinkFlowVars[i] for i in range(self.network.numSinks)) >= self.minTargetFlow,
                                  ctname="minFlow")

        # Edge opening/capacity constraints
        for i in range(self.network.numArcs):
            arc = self.network.arcsMatrix[i]
            arcID = (arc[0], arc[1], arc[2])
            ctName = "a_" + str(arcID) + "_CapAndOpen"
            self.model.add_constraint(self.arcFlowVars[i] <= self.arcOpenedVars[i] * arc[2], ctname=ctName)

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
        # TODO - REWORK FROM HERE DOWN
        # Source flow conservation
        for source in self.network.sourcesArray:
            srcObj = self.network.nodesDict[source]
            incomingEdges = srcObj.incomingEdges

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
                self.model.add_constraint(self.sourceFlowVars[int(nodeID)] ==
                                          sum(self.edgeFlowVars[i] for i in outgoingIDs),
                                          ctname=ctName)
            # Sink flow conservation
            elif nodeType == "t":
                ctName = "t" + str(nodeID) + "Conserv"
                self.model.add_constraint(self.sinkFlowVars[int(nodeID)] ==
                                          sum(self.edgeFlowVars[i] for i in incomingIDs),
                                          ctname=ctName)
            # Intermediate flow conservation
            elif nodeType == "n":
                ctName = "n" + str(nodeID) + "Conserv"
                self.model.add_constraint(sum(
                    self.edgeFlowVars[i] for i in incomingIDs) - sum(
                    self.edgeFlowVars[j] for j in outgoingIDs) == 0,
                                          ctname=ctName)
        # =================== OBJECTIVE FUNCTION ===================
        self.model.set_objective("min",
                                 sum(self.sourceFlowVars[i] * self.FCFN.nodesDict["s" + str(i)].variableCost for i
                                     in range(self.FCFN.numSources))
                                 + sum(self.sinkFlowVars[j] * self.FCFN.nodesDict["t" + str(j)].variableCost for j
                                       in range(self.FCFN.numSinks)) + sum(
                                     self.edgeFlowVars[n] * self.FCFN.edgesDict["e" + str(n)].variableCost
                                     for n
                                     in range(self.FCFN.numEdges)) + sum(
                                     self.edgeOpenedVars[m] * self.FCFN.edgesDict["e" + str(m)].fixedCost for
                                     m in
                                     range(self.FCFN.numEdges)))

    def solveModel(self) -> None:
        """Solves the MILP model in CPLEX"""
        print("\nAttempting to solve model...")
        self.model.solve()
        self.isRun = True
        print("Solver execution complete...\n")

    def writeSolution(self) -> None:
        """Writes the solution to the FCFN instance by updating output attributes across the FCFN, nodes, and edges"""
        if self.model.solution is not None:
            # Disperse solution results back to FCFN
            self.FCFN.isSolved = True
            self.FCFN.minTargetFlow = self.minTargetFlow
            self.FCFN.totalCost = self.model.solution.get_objective_value()
            self.FCFN.totalFlow = sum(self.model.solution.get_value_dict(self.sinkFlowVars))
            # Disperse solution results back to sources
            sourceValues = self.model.solution.get_value_dict(self.sourceFlowVars)
            for i in range(self.FCFN.numSources):
                thisSource = self.FCFN.nodesDict["s" + str(i)]
                if sourceValues[i] > 0:
                    thisSource.opened = True
                    thisSource.flow = sourceValues[i]
                    thisSource.totalCost = thisSource.flow * thisSource.variableCost
            # Disperse solution results back to sinks
            sinkValues = self.model.solution.get_value_dict(self.sinkFlowVars)
            for i in range(self.FCFN.numSinks):
                thisSink = self.FCFN.nodesDict["t" + str(i)]
                if sinkValues[i] > 0:
                    thisSink.opened = True
                    thisSink.flow = sinkValues[i]
                    thisSink.totalCost = thisSink.flow * thisSink.variableCost
            # Disperse solution results back to edges
            edgeValues = self.model.solution.get_value_dict(self.edgeFlowVars)
            for i in range(self.FCFN.numEdges):
                thisEdge = self.FCFN.edgesDict["e" + str(i)]
                if edgeValues[i] > 0:
                    thisEdge.opened = True
                    thisEdge.flow = edgeValues[i]
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

    def printSolverOverview(self) -> None:
        """Prints the most important and concise details of the solver, model and solution"""
        self.model.print_information()
        if self.isRun is True:
            print(self.model.get_solve_details())
            print("Solved by= " + self.model.solution.solved_by + "\n")
            self.model.print_solution()
            print("Total Cost: " + str(self.FCFN.totalCost))
            print("Total Flow: " + str(self.FCFN.totalFlow))

    def printModel(self) -> None:
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
            print(self.model.get_constraint_by_name("e" + str(i) + "CapAndOpen"))

    def printSolution(self) -> None:
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
