from docplex.mp.model import Model

from src.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class MILPsolver:
    """Class that solves a FCFN instance optimally with CPLEX 20.1"""

    def __init__(self, FCFNinstance: FixedChargeFlowNetwork, targetFlow: int):
        """Constructor of a MILP-Solver instance"""
        # Input attributes
        self.FCFN = FCFNinstance
        self.targetFlow = targetFlow
        # Output attributes
        self.solved = False
        self.totalFlow = 0
        self.totalCost = 0

    def solveFCNF(self):
        """Solves the FCNF instance via a reduction to a MILP solved in CPLEX"""
        m = Model(name='single variable')
        x = m.binary_var(name="x")
        c1 = m.add_constraint(x >= 2, ctname="const1")
        m.set_objective("min", 3 * x)
        m.print_information()
        m.solve()
        m.print_solution()
