from src.FixedCostNetworkFlowSolver.FCNF import FCNF
from src.FixedCostNetworkFlowSolver.MILP import MILP
from src.FixedCostNetworkFlowSolver.Visualize import Visualize

"""PY FILE USED TO RUN THE PARALLEL-EDGES-FCNF-SOLVER"""

# Test of the FCFN/Node/Edge Classes
FCNF = FCNF()
FCNF.loadFCFN("small")
# FCNF.printAllNodeData()
# FCNF.printAllEdgeData()
visual = Visualize(FCNF)
visual.drawGraph(FCNF.name)

# Test of the MILPsolver Class
solver = MILP(FCNF, 100)
solver.buildModel()
solver.printMILPmodel()
solver.solveModel()
solver.writeSolution()

# Test of the Visualize Class
visual = Visualize(FCNF)
visual.drawGraph(FCNF.name)
