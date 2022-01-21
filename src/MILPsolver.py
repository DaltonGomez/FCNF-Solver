from docplex.mp.model import Model

from src.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class MILPsolver:
    """Class that solves a FCFN instance optimally with CPLEX 20.1"""

    def __init__(self, FCFNinstance: FixedChargeFlowNetwork, minTargetFlow: int):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.FCFN = FCFNinstance
        self.minTargetFlow = minTargetFlow
        # Solver
        self.model = Model(name="FCFN-MILP-Solver", log_output=True, checker="full", cts_by_name=True)
        # TODO- Turn off parameters as model accuracy is ensured
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
        self.model.edgeOpenedVar = self.model.binary_var_list(self.FCFN.numEdges, name="y")
        self.model.edgeFlowVars = self.model.continuous_var_list(self.FCFN.numEdges, name="e")
        # Sets lower bound of zero on all edge flows (i.e. no backwards flow allowed)
        self.model.change_var_lower_bounds(self.model.edgeFlowVars, lbs=0)

        # =================== CONSTRAINTS ===================
        # Minimum flow constraint
        self.model.add_constraint(
            sum(self.model.sinkFlowVars[i] for i in range(self.FCFN.numSinks)) >= self.minTargetFlow,
            ctname="minFlowConstraint")

        # Edge opening constraints
        for i in range(self.FCFN.numEdges):
            edgeKey = self.FCFN.edgesMap["e" + str(i)]
            ctName = "e" + str(i) + "Open"
            edgeCapacity = self.FCFN.edgesDict[edgeKey].capacity
            self.model.add_constraint(self.model.edgeFlowVars[i] <= self.model.edgeOpenedVar[i] * edgeCapacity,
                                      ctname=ctName)

        # Capacity constraints of sources
        for i in range(self.FCFN.numSources):
            sourceKey = "s" + str(i)
            ctName = sourceKey + "Cap"
            srcCapacity = self.FCFN.nodesDict[sourceKey].capacity
            self.model.add_constraint(self.model.sourceFlowVars[i] <= srcCapacity, ctname=ctName)

        # Capacity constraints of sinks
        for i in range(self.FCFN.numSinks):
            sinkKey = "t" + str(i)
            ctName = sinkKey + "Cap"
            sinkCapacity = self.FCFN.nodesDict[sinkKey].capacity
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
            print(nodeID)
            print(outgoingIDs)
            print(incomingIDs)
            # Source flow conservation
            if nodeType == "s":
                ctName = "s" + str(nodeID) + "Conserv"
                self.model.add_constraint(self.model.sourceFlowVars[int(nodeID)] ==
                                          sum(self.model.edgeFlowVars[i] for i in outgoingIDs),
                                          ctname=ctName)
            # Sink flow conservation
            if nodeType == "t":
                ctName = "t" + str(nodeID) + "Conserv"
                self.model.add_constraint(self.model.sinkFlowVars[int(nodeID)] ==
                                          sum(self.model.edgeFlowVars[i] for i in incomingIDs),
                                          ctname=ctName)
            # Transshipment flow conservation
            if nodeType == "n":
                ctName = "n" + str(nodeID) + "Conserv"
                self.model.add_constraint(sum(self.model.edgeFlowVars[i] for i in incomingIDs) - sum(
                    self.model.edgeFlowVars[j] for j in outgoingIDs) == 0, ctname=ctName)
        # =================== OBJECTIVE FUNCTION ===================

        # TODO- Add a printModel() method
        print(self.model.get_constraint_by_name("s0Conserv"))
        print(self.model.get_constraint_by_name("s1Conserv"))
        print(self.model.get_constraint_by_name("s2Conserv"))
        print(self.model.get_constraint_by_name("t0Conserv"))
        print(self.model.get_constraint_by_name("t1Conserv"))
        print(self.model.get_constraint_by_name("n0Conserv"))
        print(self.model.get_constraint_by_name("n1Conserv"))
        print(self.model.get_constraint_by_name("n2Conserv"))
        print(self.model.get_constraint_by_name("n3Conserv"))
        print(self.model.get_constraint_by_name("n4Conserv"))
        print(self.model.get_constraint_by_name("minFlowConstraint"))
        print(self.model.get_constraint_by_name("s0Cap"))
        print(self.model.get_constraint_by_name("s1Cap"))
        print(self.model.get_constraint_by_name("s2Cap"))
        print(self.model.get_constraint_by_name("t0Cap"))
        print(self.model.get_constraint_by_name("t1Cap"))
        print(self.model.get_constraint_by_name("e0Open"))
        print(self.model.get_constraint_by_name("e1Open"))
        print(self.model.get_constraint_by_name("e2Open"))
        print(self.model.get_constraint_by_name("e3Open"))
        print(self.model.get_constraint_by_name("e4Open"))
        print(self.model.get_constraint_by_name("e5Open"))
        print(self.model.get_constraint_by_name("e6Open"))
        print(self.model.get_constraint_by_name("e7Open"))
        print(self.model.get_constraint_by_name("e8Open"))
        print(self.model.get_constraint_by_name("e9Open"))
        print(self.model.get_constraint_by_name("e10Open"))
        print(self.model.get_constraint_by_name("e11Open"))
        print(self.model.get_constraint_by_name("e12Open"))

    def testCPLEX(self):
        """Solves the FCNF instance via a reduction to a MILP solved in CPLEX"""
        model = Model(name="test")
        x = model.binary_var(name="x")
        c1 = model.add_constraint(x >= 2, ctname="const1")
        model.set_objective("min", 3 * x)
        model.print_information()
        model.solve()
        model.print_solution()
