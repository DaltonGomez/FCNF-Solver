from src.FixedChargeFlowNetwork import FixedChargeFlowNetwork
from src.MILPsolver import MILPsolver
from src.Visualize import Visualize

"""PY FILE USED TO RUN THE FCNF-SOLVER"""

# Test of the FCFN/Node/Edge Classes
FCFN = FixedChargeFlowNetwork()
FCFN.loadFCFN("small")
# FCFN.printAllNodeData()
# FCFN.printAllEdgeData()

# Test of the MILPsolver Class
solver = MILPsolver(FCFN, 25)
solver.buildModel()
solver.printMILPmodel()
solver.solveModel()
solver.writeSolution()

# Test of the Visualize Class
visual = Visualize(FCFN)
# visual.drawGraphUiOptions(FCFN.name)
visual.drawGraphHardcodeOptionsTwo(FCFN.name)
