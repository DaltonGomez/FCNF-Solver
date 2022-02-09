from src.FixedChargeNetwork.ExactSolver import ExactSolver
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""PY FILE USED TO RUN THE PARALLEL-EDGES-FCNF-SOLVER"""

# Test of the FCFN/Node/Edge Classes
FCFNinstance = FixedChargeFlowNetwork()
FCFNinstance.loadFCFNfromDisc("mediumOneCap")

# Test of the MILPsolver Class
solver = ExactSolver(FCFNinstance, 125)
solver.buildModel()
solver.printModel()
solver.solveModel()
solver.writeSolution()
# Test of the Visualize Class
# visual = Visualize(FCFNinstance)
# visual.drawGraph(FCFNinstance.name)

# Test of GA Population
# population = GeneticPopulation(FCFNinstance, 125, 100, 10)
# population.evolvePopulation()

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
