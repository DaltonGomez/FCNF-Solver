from docplex.mp.model import Model

from src.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class MILPsolver:
    """Class that solves a FCFN instance optimally with CPLEX 20.1"""

    def __init__(self, FCFNinstance: FixedChargeFlowNetwork, minFlowConstraint: int):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.FCFN = FCFNinstance
        self.minFlowConstraint = minFlowConstraint
        # Solver
        self.model = Model(name="FCFN-MILP-Solver", log_output=True, checker="full", cts_by_name=True)
        # TODO- Turn off parameters as model accuracy is ensured
        # Output attributes
        self.solved = False
        self.totalFlow = 0
        self.totalCost = 0

    def buildModel(self):
        """Builds the decision variables, constraints, and object function of the MILP model from the FCFN instance"""
        Y = self.model.binary_var_list(self.FCFN.numEdges, name="y")
        print(Y)

    def testCPLEX(self):
        """Solves the FCNF instance via a reduction to a MILP solved in CPLEX"""
        model = Model(name="test")
        x = model.binary_var(name="x")
        c1 = model.add_constraint(x >= 2, ctname="const1")
        model.set_objective("min", 3 * x)
        model.print_information()
        model.solve()
        model.print_solution()
