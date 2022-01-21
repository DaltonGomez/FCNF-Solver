from docplex.mp.model import Model

from src.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class MILPsolver:
    """Class that solves a FCFN instance optimally with CPLEX 20.1"""

    def __init__(self, FCFNinstance: FixedChargeFlowNetwork, minTargetFlow: int):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.FCFN = FCFNinstance
        self.minTargetFlow = minTargetFlow
        # Solver model
        self.model = Model(name="FCFN-MILP-Solver", log_output=False, cts_by_name=True)
        # Output attributes
        self.solved = False
        self.totalFlow = 0
        self.totalCost = 0

    def buildModel(self):
        """Builds the decision variables, constraints, and object function of the MILP model from the FCFN instance"""
        # =================== DECISION VARIABLES ===================
        # Source and sink decision variables
        self.model.sourceFlowVars = self.model.continuous_var_list(self.FCFN.numSources, name="s")
        self.model.sinkFlowVars = self.model.continuous_var_list(self.FCFN.numSinks, name="t")
        # Edge decision variables
        self.model.edgeOpenedVars = self.model.binary_var_list(self.FCFN.numEdges, name="y")
        self.model.edgeFlowVars = self.model.continuous_var_list(self.FCFN.numEdges, name="e")
        # Sets lower bound of zero on all edge flows (i.e. no backwards flow allowed)
        self.model.change_var_lower_bounds(self.model.edgeFlowVars, lbs=0)

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint
        self.model.add_constraint(
            sum(self.model.sinkFlowVars[i] for i in range(self.FCFN.numSinks)) >= self.minTargetFlow,
            ctname="minFlow")

        # Edge opening/capacity constraints
        for i in range(self.FCFN.numEdges):
            ctName = "e" + str(i) + "CapAndOpen"
            edgeCapacity = self.FCFN.edgesDict["e" + str(i)].capacity
            self.model.add_constraint(self.model.edgeFlowVars[i] <= self.model.edgeOpenedVars[i] * edgeCapacity,
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
                self.model.add_constraint(sum(self.model.edgeFlowVars[i] for i in incomingIDs) - sum(
                    self.model.edgeFlowVars[j] for j in outgoingIDs) == 0, ctname=ctName)
        # =================== OBJECTIVE FUNCTION ===================
        self.model.set_objective("min",
                                 sum(self.model.sourceFlowVars[i] * self.FCFN.nodesDict["s" + str(i)].variableCost for i
                                     in range(self.FCFN.numSources))
                                 + sum(self.model.sinkFlowVars[j] * self.FCFN.nodesDict["t" + str(j)].variableCost for j
                                       in range(self.FCFN.numSinks)) + sum(
                                     self.model.edgeFlowVars[k] * self.FCFN.edgesDict["e" + str(k)].variableCost for k
                                     in range(self.FCFN.numEdges)) + sum(
                                     self.model.edgeOpenedVars[m] * self.FCFN.edgesDict["e" + str(m)].fixedCost for m in
                                     range(self.FCFN.numEdges)))

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
            self.FCFN.solved = True
            self.FCFN.minTargetFlow = self.minTargetFlow
            self.FCFN.totalCost = self.totalCost
            self.FCFN.totalFlow = self.totalFlow
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
            edgeValues = self.model.solution.get_value_list(self.model.edgeFlowVars)
            for i in range(self.FCFN.numEdges):
                thisEdge = self.FCFN.edgesDict["e" + str(i)]
                if edgeValues[i] > 0:
                    thisEdge.opened = True
                    thisEdge.flow = edgeValues[i]
                    thisEdge.totalCost = thisEdge.flow * thisEdge.variableCost + thisEdge.fixedCost
            # Disperse solution results back to transshipment nodes
            for i in range(self.FCFN.numNodes - (self.FCFN.numSources + self.FCFN.numSinks)):
                thisNode = self.FCFN.nodesDict["n" + str(i)]
                for edge in thisNode.incomingEdges:
                    thisNode.flow += self.FCFN.edgesDict[edge].flow
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
