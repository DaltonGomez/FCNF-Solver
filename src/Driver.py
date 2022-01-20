from src.FixedChargeFlowNetwork import FixedChargeFlowNetwork
from src.MILPsolver import MILPsolver

"""PY FILE USED TO RUN THE FCNF-SOLVER"""

# Test Driver
FCFN = FixedChargeFlowNetwork()
FCFN.loadFCFN("small")
FCFN.drawFCNF()
solver = MILPsolver(FCFN, 70)
solver.solveFCNF()
