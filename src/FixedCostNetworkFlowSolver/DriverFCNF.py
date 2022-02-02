from src.AlphaReducedFCNF.GeneticPopulation import GeneticPopulation
from src.FixedCostNetworkFlowSolver.FCNF import FCNF
from src.FixedCostNetworkFlowSolver.MILP import MILP
from src.FixedCostNetworkFlowSolver.Visualize import Visualize

"""PY FILE USED TO RUN THE PARALLEL-EDGES-FCNF-SOLVER"""

# Test of the FCFN/Node/Edge Classes
FCNFinstance = FCNF()
FCNFinstance.loadFCFN("smallOneCap")

# Test of GA Population
population = GeneticPopulation(FCNFinstance, 36, 5, 5)
population.evolvePopulation()

# Test of the MILPsolver Class
solver = MILP(FCNFinstance, 36)
solver.buildModel()
solver.printMILPmodel()
solver.solveModel()
solver.writeSolution()
# Test of the Visualize Class
visual = Visualize(FCNFinstance)
visual.drawGraph(FCNFinstance.name)

# Test of alpha relaxed class
# alphaFCNF = AlphaFCNF(FCNFinstance)
# print(alphaFCNF.alphaValues)
# solverLP = AlphaLP(alphaFCNF, 36)
# solverLP.buildModel()
# solverLP.solveModel()
# solverLP.writeSolution()
# alphaFCNF.calculateTrueCost()
# visualAlpha = AlphaVisualize(alphaFCNF)
# visualAlpha.drawGraph(alphaFCNF.name)
