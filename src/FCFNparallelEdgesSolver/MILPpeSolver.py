from docplex.mp.model import Model

from src.FCFNparallelEdgesSolver.FCFNparallelEdges import FCFNparallelEdges


class MILPpeSolver:
    """Class that solves a FCFN instance w/ parallel edges optimally with CPLEX 20.1"""

    def __init__(self, FCFNpeInstance: FCFNparallelEdges, minTargetFlow: int):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.FCFNpe = FCFNpeInstance
        self.minTargetFlow = minTargetFlow
        # Solver model
        self.model = Model(name="FCFNpe-MILP-Solver", log_output=False, cts_by_name=True)
        # Output attributes
        self.solved = False
        self.totalFlow = 0
        self.totalCost = 0

    def buildModel(self):
        """Builds the decision variables, constraints, and object function of the MILP model from the FCFN instance"""
        # =================== DECISION VARIABLES ===================
        # Source and sink decision variables
        self.model.sourceFlowVars = self.model.continuous_var_list(self.FCFNpe.numSources, name="s")
        self.model.sinkFlowVars = self.model.continuous_var_list(self.FCFNpe.numSinks, name="t")
        # Edge decision variables - Built as a matrix indexed on [edge][capacity] and with a lower bound of 0
        self.model.edgeOpenedVars = self.model.binary_var_matrix(self.FCFNpe.numEdges, self.FCFNpe.numEdgeCaps,
                                                                 name="y")
        self.model.edgeFlowVars = self.model.continuous_var_matrix(self.FCFNpe.numEdges, self.FCFNpe.numEdgeCaps,
                                                                   name="e", lb=0)

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint
        self.model.add_constraint(
            sum(self.model.sinkFlowVars[i] for i in range(self.FCFNpe.numSinks)) >= self.minTargetFlow,
            ctname="minFlow")

        # Edge opening/capacity constraints
        for i in range(self.FCFNpe.numEdges):
            for j in range(self.FCFNpe.numEdgeCaps):
                ctName = "e" + str(i) + "_" + str(j) + "CapAndOpen"
                edgeCapacity = int(self.FCFNpe.edgeCaps[j])
                self.model.add_constraint(
                    self.model.edgeFlowVars[(i, j)] <= self.model.edgeOpenedVars[(i, j)] * edgeCapacity,
                    ctname=ctName)

        # Capacity constraints of sources
        for i in range(self.FCFNpe.numSources):
            ctName = "s" + str(i) + "Cap"
            srcCapacity = self.FCFNpe.nodesDict["s" + str(i)].capacity
            self.model.add_constraint(self.model.sourceFlowVars[i] <= srcCapacity, ctname=ctName)

        # Capacity constraints of sinks
        for i in range(self.FCFNpe.numSinks):
            ctName = "t" + str(i) + "Cap"
            sinkCapacity = self.FCFNpe.nodesDict["t" + str(i)].capacity
            self.model.add_constraint(self.model.sinkFlowVars[i] <= sinkCapacity, ctname=ctName)

        # Only one capacity per edge constraints
        for i in range(self.FCFNpe.numEdges):
            ctName = "e" + str(i) + "CapPerEdge"
            self.model.add_constraint(
                sum(self.model.edgeOpenedVars[(i, j)] for j in range(self.FCFNpe.numEdgeCaps)) <= 1,
                ctname=ctName)

        # Conservation of flow constraints
        for node in self.FCFNpe.nodesDict:
            nodeObj = self.FCFNpe.nodesDict[node]
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
                                              range(self.FCFNpe.numEdgeCaps)),
                                          ctname=ctName)
            # Sink flow conservation
            elif nodeType == "t":
                ctName = "t" + str(nodeID) + "Conserv"
                self.model.add_constraint(self.model.sinkFlowVars[int(nodeID)] ==
                                          sum(self.model.edgeFlowVars[(i, j)] for i in incomingIDs for j in
                                              range(self.FCFNpe.numEdgeCaps)),
                                          ctname=ctName)
            # Transshipment flow conservation
            elif nodeType == "n":
                ctName = "n" + str(nodeID) + "Conserv"
                self.model.add_constraint(sum(
                    self.model.edgeFlowVars[(i, j)] for i in incomingIDs for j in range(self.FCFNpe.numEdgeCaps)) - sum(
                    self.model.edgeFlowVars[(m, n)] for m in outgoingIDs for n in range(self.FCFNpe.numEdgeCaps)) == 0,
                                          ctname=ctName)
        # =================== OBJECTIVE FUNCTION ===================
        self.model.set_objective("min",
                                 sum(self.model.sourceFlowVars[i] * self.FCFNpe.nodesDict["s" + str(i)].variableCost for
                                     i
                                     in range(self.FCFNpe.numSources))
                                 + sum(
                                     self.model.sinkFlowVars[j] * self.FCFNpe.nodesDict["t" + str(j)].variableCost for j
                                     in range(self.FCFNpe.numSinks)) + sum(
                                     self.model.edgeFlowVars[(m, n)] * int(self.FCFNpe.edgeVariableCosts[n]) for m
                                     in range(self.FCFNpe.numEdges) for n in range(self.FCFNpe.numEdgeCaps)) + sum(
                                     self.model.edgeOpenedVars[(a, b)] * int(self.FCFNpe.edgeFixedCosts[b]) for a in
                                     range(self.FCFNpe.numEdges) for b in range(self.FCFNpe.numEdgeCaps)))

    def solveModel(self):
        """Solves the MILP model in CPLEX"""
        print("\nSolving model...")
        self.model.solve()
        self.solved = True
        print("\nModel solved!!!")

    def writeSolution(self):
        """Prints the solution to the console and updates the FCFN elements with their solution values"""
        print("=============== SOLUTION DETAILS ========================")
        print(self.model.get_solve_details())
        if self.model.solution is not None:
            print("Solved by= " + self.model.solution.solved_by + "\n")
            print("=============== SOLUTION VALUES ========================")
            self.model.print_solution()
            self.totalCost = self.model.solution.get_objective_value()
            self.totalFlow = sum(self.model.solution.get_value_list(self.model.sinkFlowVars))
            print("Total Flow: " + str(self.totalFlow))
            # Disperse solution results back to FCFN
            self.FCFNpe.solved = True
            self.FCFNpe.minTargetFlow = self.minTargetFlow
            self.FCFNpe.totalCost = self.totalCost
            self.FCFNpe.totalFlow = self.totalFlow
            # Disperse solution results back to sources
            sourceValues = self.model.solution.get_value_list(self.model.sourceFlowVars)
            for i in range(self.FCFNpe.numSources):
                thisSource = self.FCFNpe.nodesDict["s" + str(i)]
                if sourceValues[i] > 0:
                    thisSource.opened = True
                    thisSource.flow = sourceValues[i]
                    thisSource.totalCost = thisSource.flow * thisSource.variableCost
            # Disperse solution results back to sources
            sinkValues = self.model.solution.get_value_list(self.model.sinkFlowVars)
            for i in range(self.FCFNpe.numSinks):
                thisSink = self.FCFNpe.nodesDict["t" + str(i)]
                if sinkValues[i] > 0:
                    thisSink.opened = True
                    thisSink.flow = sinkValues[i]
                    thisSink.totalCost = thisSink.flow * thisSink.variableCost
            # Disperse solution results back to edges
            edgeValues = self.model.solution.get_value_dict(self.model.edgeFlowVars)
            for i in range(self.FCFNpe.numEdges):
                thisEdge = self.FCFNpe.edgesDict["e" + str(i)]
                for j in range(self.FCFNpe.numEdgeCaps):
                    if edgeValues[(i, j)] > 0:
                        thisEdge.opened = True
                        thisEdge.capacity = self.FCFNpe.edgeCaps[j]
                        thisEdge.fixedCost = self.FCFNpe.edgeFixedCosts[j]
                        thisEdge.variableCost = self.FCFNpe.edgeVariableCosts[j]
                        thisEdge.flow = edgeValues[(i, j)]
                        thisEdge.totalCost = thisEdge.flow * int(thisEdge.variableCost) + int(thisEdge.fixedCost)
            # Disperse solution results back to transshipment nodes
            for i in range(self.FCFNpe.numNodes - (self.FCFNpe.numSources + self.FCFNpe.numSinks)):
                thisNode = self.FCFNpe.nodesDict["n" + str(i)]
                for edge in thisNode.incomingEdges:
                    thisNode.flow += self.FCFNpe.edgesDict[edge].flow
                if thisNode.flow > 0:
                    thisNode.opened = True
        else:
            print("No feasible solution exists!")

    def printMILPmodel(self):
        """Prints all constraints of the MILP model for the FCFN instance (FOR DEBUGGING- DO NOT CALL ON LARGE INPUTS)"""
        print("=============== MILP MODEL OF FCFN INSTANCE ========================")
        self.model.print_information()
        print("=============== OBJECTIVE FUNCTION ========================")
        print(self.model.get_objective_expr())
        print("=============== CONSTRAINTS ========================")
        print(self.model.get_constraint_by_name("minFlow"))
        for i in range(self.FCFNpe.numSources):
            print(self.model.get_constraint_by_name("s" + str(i) + "Conserv"))
            print(self.model.get_constraint_by_name("s" + str(i) + "Cap"))
        for i in range(self.FCFNpe.numSinks):
            print(self.model.get_constraint_by_name("t" + str(i) + "Conserv"))
            print(self.model.get_constraint_by_name("t" + str(i) + "Cap"))
        for i in range(self.FCFNpe.numSinks):
            print(self.model.get_constraint_by_name("t" + str(i) + "Conserv"))
            print(self.model.get_constraint_by_name("t" + str(i) + "Cap"))
        for i in range(self.FCFNpe.numNodes - (self.FCFNpe.numSources + self.FCFNpe.numSinks)):
            print(self.model.get_constraint_by_name("n" + str(i) + "Conserv"))
        for i in range(self.FCFNpe.numEdges):
            print(self.model.get_constraint_by_name("e" + str(i) + "CapPerEdge"))
            for j in range(self.FCFNpe.numEdgeCaps):
                print(self.model.get_constraint_by_name("e" + str(i) + "_" + str(j) + "CapAndOpen"))
