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
        # Source and sink decision variables
        self.model.sourceFlowVars = self.model.continuous_var_list(self.FCFN.numSources, name="s")
        self.model.sinkFlowVars = self.model.continuous_var_list(self.FCFN.numSinks, name="t")
        # Edge decision variables
        self.model.edgeOpenedVar = self.model.binary_var_list(self.FCFN.numEdges, name="y")
        self.model.edgeFlowVars = self.model.continuous_var_list(self.FCFN.numEdges, name="e")

        # Minimum flow constraint
        self.model.add_constraint(
            sum(self.model.sinkFlowVars[i] for i in range(self.FCFN.numSinks)) >= self.minTargetFlow,
            ctname="minFlowConstraint")

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

        # TODO- Add a print model in English method
        print(self.model.get_constraint_by_name("minFlowConstraint"))
        print(self.model.get_constraint_by_name("s0Cap"))
        print(self.model.get_constraint_by_name("s1Cap"))
        print(self.model.get_constraint_by_name("s2Cap"))
        print(self.model.get_constraint_by_name("t0Cap"))
        print(self.model.get_constraint_by_name("t1Cap"))

    def testCPLEX(self):
        """Solves the FCNF instance via a reduction to a MILP solved in CPLEX"""
        model = Model(name="test")
        x = model.binary_var(name="x")
        c1 = model.add_constraint(x >= 2, ctname="const1")
        model.set_objective("min", 3 * x)
        model.print_information()
        model.solve()
        model.print_solution()
