from docplex.mp.model import Model

from src.AlphaGeneticSolver.AlphaIndividual import AlphaIndividual


class AlphaSolver:
    """Class that solves an alpha-relaxed FCFN instance w/ via a LP model within CPLEX"""

    def __init__(self, FCFN, minTargetFlow: int):
        """Constructor of an AlphaSolver instance
        NOTE: One solver is assigned to the entire population and the model constraints are built at initialization."""
        # Input attributes
        self.FCFN = FCFN
        self.minTargetFlow = minTargetFlow

        # Solver Model and State Variables
        self.model = Model(name="AlphaFCFN-LP-RelaxedSolver", log_output=False, cts_by_name=True)
        self.hasDecisionVariables = False
        self.hasConstraints = False
        self.hasObjectiveFunction = False
        self.isRun = False
        self.hasSolution = False

        # Construct the decision variables and constraints for the model once at initialization (NOTE: Time Consuming!)
        self.initializeDecisionVariables()
        self.initializeConstraints()

    def initializeDecisionVariables(self):
        """Constructs the decision variables for the LP model"""
        # =================== DECISION VARIABLES ===================
        # Source, sink, and edge decision variables
        self.model.sourceFlowVars = self.model.continuous_var_list(self.FCFN.numSources, name="s")
        self.model.sinkFlowVars = self.model.continuous_var_list(self.FCFN.numSinks, name="t")
        self.model.edgeFlowVars = self.model.continuous_var_list(self.FCFN.numEdges, name="e", lb=0)
        self.hasDecisionVariables = True

    def initializeConstraints(self):
        """Constructs the constraints for the LP model"""
        # =================== CONSTRAINTS ===================
        # Minimum flow constraint (Constructed as the sum of all sinks in-flows)
        self.model.add_constraint(
            sum(self.model.sinkFlowVars[i] for i in range(self.FCFN.numSinks)) >= self.minTargetFlow,
            ctname="minFlow")

        # Edge opening/capacity constraints
        for i in range(self.FCFN.numEdges):
            ctName = "e" + str(i) + "Cap"
            edgeCapacity = self.FCFN.edgesDict["e" + str(i)].capacity
            self.model.add_constraint(self.model.edgeFlowVars[i] <= edgeCapacity, ctname=ctName)

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
            # Intermediate node flow conservation
            elif nodeType == "n":
                ctName = "n" + str(nodeID) + "Conserv"
                self.model.add_constraint(sum(
                    self.model.edgeFlowVars[i] for i in incomingIDs) - sum(
                    self.model.edgeFlowVars[j] for j in outgoingIDs) == 0,
                                          ctname=ctName)
            self.hasConstraints = True

    def updateObjectiveFunction(self, alphaValues: list):
        """Deletes any previous objective function and writes a new objective function using the input alpha values"""
        if self.model.has_objective() or self.model.solution is not None:
            print("TEST")
            self.resetSolver()
        # =================== OBJECTIVE FUNCTION ===================
        self.model.set_objective("min", sum(
            self.model.sourceFlowVars[i] * self.FCFN.nodesDict["s" + str(i)].variableCost for i in
            range(self.FCFN.numSources))
                                 + sum(
            self.model.sinkFlowVars[j] * self.FCFN.nodesDict["t" + str(j)].variableCost for j in
            range(self.FCFN.numSinks))
                                 + sum(self.model.edgeFlowVars[n] * (self.FCFN.edgesDict["e" + str(n)].variableCost
                                                                     + alphaValues[n] * self.FCFN.edgesDict[
                                                                         "e" + str(n)].fixedCost) for n in
                                       range(self.FCFN.numEdges)))
        self.hasObjectiveFunction = True

    def solveModel(self) -> None:
        """Solves the alpha-relaxed LP model in CPLEX"""
        # print("Attempting to solve model...")
        self.model.solve()
        self.isRun = True
        # print("Solver execution complete...\n")

    def writeSolution(self, individual: AlphaIndividual) -> None:
        """Writes the solution to the individual by updating output variables in the openedNodes and openedEdges dicts"""
        if self.model.solution is not None:
            self.hasSolution = True
            # Disperse solution results back to individual
            individual.isSolved = True
            individual.minTargetFlow = self.minTargetFlow
            individual.fakeCost = self.model.solution.get_objective_value()
            individual.totalFlow = sum(self.model.solution.get_value_list(self.model.sinkFlowVars))
            # CONSTRUCT OPENED NODES DICT
            # Extract source values
            sourceValues = self.model.solution.get_value_list(self.model.sourceFlowVars)
            for i in range(individual.FCFN.numSources):
                thisSource = individual.FCFN.nodesDict["s" + str(i)]
                if sourceValues[i] > 0:
                    flow = sourceValues[i]
                    totalCost = flow * thisSource.variableCost
                    individual.openedNodesDict["s" + str(i)] = (flow, totalCost)
            # Extract sink values
            sinkValues = self.model.solution.get_value_list(self.model.sinkFlowVars)
            for i in range(individual.FCFN.numSinks):
                thisSink = individual.FCFN.nodesDict["t" + str(i)]
                if sinkValues[i] > 0:
                    flow = sinkValues[i]
                    totalCost = flow * thisSink.variableCost
                    individual.openedNodesDict["t" + str(i)] = (flow, totalCost)
            # Extract edge values
            edgeValues = self.model.solution.get_value_list(self.model.edgeFlowVars)
            for i in range(individual.FCFN.numEdges):
                thisEdge = individual.FCFN.edgesDict["e" + str(i)]
                if edgeValues[i] > 0:
                    flow = edgeValues[i]
                    totalCost = flow * thisEdge.variableCost + thisEdge.fixedCost
                    individual.openedEdgesDict["e" + str(i)] = (flow, totalCost)
            # Extract intermediate node values
            for i in range(individual.FCFN.numIntermediateNodes):
                thisNode = individual.FCFN.nodesDict["n" + str(i)]
                flow = 0
                for edge in thisNode.incomingEdges:
                    edgeID = int(edge.lstrip("e"))
                    flow += edgeValues[edgeID]
                    if flow > 0:
                        individual.openedNodesDict["n" + str(i)] = (flow, 0)
        else:
            print("No feasible solution exists!")

    def resetSolver(self):
        """Resets the solver to its initialized state (i.e. only variables and constraints)"""
        self.model.remove_objective()
        self.hasObjectiveFunction = False
        self.model.solution.clear()
        self.hasSolution = False
        self.isRun = False

    def printCurrentSolverDetails(self) -> None:
        """Prints the most important and concise details of the solver, model and solution"""
        self.model.print_information()
        if self.isRun is True:
            print(self.model.get_solve_details())
            print("Solved by= " + self.model.solution.solved_by + "\n")
            self.model.print_solution()

    def printCurrentModel(self) -> None:
        """Prints all constraints of the alpha-relaxed LP model for the Individual instance (FOR DEBUGGING- DON'T CALL ON LARGE INPUTS)"""
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
        for i in range(self.FCFN.numIntermediateNodes):
            print(self.model.get_constraint_by_name("n" + str(i) + "Conserv"))
        for i in range(self.FCFN.numEdges):
            print(self.model.get_constraint_by_name("e" + str(i) + "Cap"))

    def printCurrentSolution(self) -> None:
        """Prints the solution data of the Individual instance solved by the alpha-relaxed LP model"""
        print("=============== SOLUTION DETAILS ========================")
        print(self.model.get_solve_details())
        if self.hasSolution is True:
            print("Solved by= " + self.model.solution.solved_by + "\n")
            print("=============== SOLUTION VALUES ========================")
            self.model.print_solution()
        else:
            print("No feasible solution exists!")
