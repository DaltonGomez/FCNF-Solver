from docplex.mp.model import Model

from src.AlphaReducedFCNF.AlphaFCNF import AlphaFCNF


class AlphaLP:
    """Class that solves an alpha-reduced FCNF instance w/ via a LP model within CPLEX 20.1"""

    # TODO - Revise to account for parallel edges

    def __init__(self, alphaFCNFinstance: AlphaFCNF, minTargetFlow: int):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.alphaFCNF = alphaFCNFinstance
        self.minTargetFlow = minTargetFlow
        # Solver model
        self.model = Model(name="alphaFCFN-LP-Solver", log_output=False, cts_by_name=True)
        # Output attributes
        self.solved = False
        self.totalFlow = 0
        self.totalCost = 0

    def buildModel(self):
        """Builds the decision variables, constraints, and object function of the MILP model from the FCFN instance"""
        # =================== DECISION VARIABLES ===================
        # Source and sink decision variables
        self.model.sourceFlowVars = self.model.continuous_var_list(self.alphaFCNF.FCNF.numSources, name="s")
        self.model.sinkFlowVars = self.model.continuous_var_list(self.alphaFCNF.FCNF.numSinks, name="t")
        self.model.edgeFlowVars = self.model.continuous_var_list(self.alphaFCNF.FCNF.numEdges, name="e", lb=0)

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint
        self.model.add_constraint(
            sum(self.model.sinkFlowVars[i] for i in range(self.alphaFCNF.FCNF.numSinks)) >= self.minTargetFlow,
            ctname="minFlow")

        # Edge capacity constraints
        for i in range(self.alphaFCNF.FCNF.numEdges):
            ctName = "e" + str(i) + "CapAndOpen"
            edgeCapacity = int(self.alphaFCNF.FCNF.edgeCaps[0])
            self.model.add_constraint(self.model.edgeFlowVars[i] <= edgeCapacity, ctname=ctName)

        # Capacity constraints of sources
        for i in range(self.alphaFCNF.FCNF.numSources):
            ctName = "s" + str(i) + "Cap"
            srcCapacity = self.alphaFCNF.FCNF.nodesDict["s" + str(i)].capacity
            self.model.add_constraint(self.model.sourceFlowVars[i] <= srcCapacity, ctname=ctName)

        # Capacity constraints of sinks
        for i in range(self.alphaFCNF.FCNF.numSinks):
            ctName = "t" + str(i) + "Cap"
            sinkCapacity = self.alphaFCNF.FCNF.nodesDict["t" + str(i)].capacity
            self.model.add_constraint(self.model.sinkFlowVars[i] <= sinkCapacity, ctname=ctName)

        # Conservation of flow constraints
        for node in self.alphaFCNF.FCNF.nodesDict:
            nodeObj = self.alphaFCNF.FCNF.nodesDict[node]
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
                                          sum(self.model.edgeFlowVars[i] for i in outgoingIDs), ctname=ctName)
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
                    self.model.edgeFlowVars[m] for m in outgoingIDs) == 0,
                                          ctname=ctName)
        # =================== OBJECTIVE FUNCTION ===================
        self.model.set_objective("min",
                                 sum(self.model.sourceFlowVars[i] * self.alphaFCNF.FCNF.nodesDict[
                                     "s" + str(i)].variableCost for
                                     i
                                     in range(self.alphaFCNF.FCNF.numSources))
                                 + sum(
                                     self.model.sinkFlowVars[j] * self.alphaFCNF.FCNF.nodesDict[
                                         "t" + str(j)].variableCost for j
                                     in range(self.alphaFCNF.FCNF.numSinks)) + sum(
                                     self.model.edgeFlowVars[m] * (int(self.alphaFCNF.FCNF.edgeVariableCosts[0]) +
                                                                   self.alphaFCNF.alphaValues[m] * int(
                                                 self.alphaFCNF.FCNF.edgeFixedCosts[0])) for m
                                     in range(self.alphaFCNF.FCNF.numEdges)))

    def solveModel(self):
        """Solves the alpha-relaxed LP model in CPLEX"""
        print("\nSolving model...")
        self.model.solve()
        self.solved = True
        print("\nModel solved!!!")

    def writeSolution(self):
        """Prints the solution to the console and updates the FCNF elements with their solution values"""
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
            self.alphaFCNF.solved = True
            self.alphaFCNF.minTargetFlow = self.minTargetFlow
            self.alphaFCNF.totalCost = self.totalCost
            self.alphaFCNF.totalFlow = self.totalFlow
            # Disperse solution results back to sources
            sourceValues = self.model.solution.get_value_list(self.model.sourceFlowVars)
            for i in range(self.alphaFCNF.FCNF.numSources):
                thisSource = self.alphaFCNF.FCNF.nodesDict["s" + str(i)]
                if sourceValues[i] > 0:
                    thisSource.opened = True
                    thisSource.flow = sourceValues[i]
                    thisSource.totalCost = thisSource.flow * thisSource.variableCost
            # Disperse solution results back to sources
            sinkValues = self.model.solution.get_value_list(self.model.sinkFlowVars)
            for i in range(self.alphaFCNF.FCNF.numSinks):
                thisSink = self.alphaFCNF.FCNF.nodesDict["t" + str(i)]
                if sinkValues[i] > 0:
                    thisSink.opened = True
                    thisSink.flow = sinkValues[i]
                    thisSink.totalCost = thisSink.flow * thisSink.variableCost
            # Disperse solution results back to edges
            edgeValues = self.model.solution.get_value_list(self.model.edgeFlowVars)
            for i in range(self.alphaFCNF.FCNF.numEdges):
                thisEdge = self.alphaFCNF.FCNF.edgesDict["e" + str(i)]
                if edgeValues[i] > 0:
                    thisEdge.opened = True
                    thisEdge.capacity = self.alphaFCNF.FCNF.edgeCaps[0]
                    thisEdge.fixedCost = self.alphaFCNF.FCNF.edgeFixedCosts[0]
                    thisEdge.variableCost = self.alphaFCNF.FCNF.edgeVariableCosts[0]
                    thisEdge.flow = edgeValues[i]
                    thisEdge.totalCost = thisEdge.flow * int(thisEdge.variableCost) + int(thisEdge.fixedCost)
            # Disperse solution results back to transshipment nodes
            for i in range(
                    self.alphaFCNF.FCNF.numNodes - (self.alphaFCNF.FCNF.numSources + self.alphaFCNF.FCNF.numSinks)):
                thisNode = self.alphaFCNF.FCNF.nodesDict["n" + str(i)]
                for edge in thisNode.incomingEdges:
                    thisNode.flow += self.alphaFCNF.FCNF.edgesDict[edge].flow
                if thisNode.flow > 0:
                    thisNode.opened = True
        else:
            print("No feasible solution exists!")
