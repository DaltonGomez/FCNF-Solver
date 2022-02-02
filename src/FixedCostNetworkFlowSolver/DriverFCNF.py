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

# Test of alpha relaxed class
alphaFCNF = AlphaFCNF(FCNFinstance)
print(alphaFCNF.alphaValues)
solverLP = AlphaLP(alphaFCNF, 35)
solverLP.buildModel()
solverLP.solveModel()
solverLP.writeSolution()
alphaFCNF.calculateTrueCost()
visualAlpha = AlphaVisualize(alphaFCNF)
visualAlpha.drawGraph(alphaFCNF.name)

# Test of the MILPsolver Class
solver = MILP(FCNFinstance, 35)
solver.buildModel()
solver.printMILPmodel()
solver.solveModel()
solver.writeSolution()
# Test of the Visualize Class
visual = Visualize(FCNFinstance)
visual.drawGraph(FCNFinstance.name)
