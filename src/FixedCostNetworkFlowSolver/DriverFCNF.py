from src.AlphaReducedFCNF.AlphaFCNF import AlphaFCNF
from src.AlphaReducedFCNF.AlphaLP import AlphaLP
from src.AlphaReducedFCNF.AlphaVisualize import AlphaVisualize
from src.FixedCostNetworkFlowSolver.FCNF import FCNF
from src.FixedCostNetworkFlowSolver.MILP import MILP
from src.FixedCostNetworkFlowSolver.Visualize import Visualize

"""PY FILE USED TO RUN THE PARALLEL-EDGES-FCNF-SOLVER"""

# Test of the FCFN/Node/Edge Classes
FCNFinstance = FCNF()
FCNFinstance.loadFCFN("smallOneCap")
# FCNF.printAllNodeData()
# FCNF.printAllEdgeData()
# visual = Visualize(FCNF)
# visual.drawGraph(FCNF.name)


# Test of the MILPsolver Class
solver = MILP(FCNFinstance, 35)
solver.buildModel()
solver.printMILPmodel()
solver.solveModel()
solver.writeSolution()

# Test of the Visualize Class
visual = Visualize(FCNFinstance)
visual.drawGraph(FCNFinstance.name)
# FCNF.printAllNodeData()
# FCNF.printAllEdgeData()

alphaFCNF = AlphaFCNF(FCNFinstance)
alphaFCNF.resetOriginalFCNFSolution()
solverLP = AlphaLP(alphaFCNF, 35)
solverLP.buildModel()
solverLP.solveModel()
solverLP.writeSolution()
visualAlpha = AlphaVisualize(alphaFCNF)
visualAlpha.drawGraph(alphaFCNF.name)
print(alphaFCNF.alphaValues)
