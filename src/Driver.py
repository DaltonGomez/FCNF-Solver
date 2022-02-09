from src.AlphaGeneticSolver.AlphaIndividual import AlphaIndividual
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# Test of the FCFN
FCFNinstance = FixedChargeFlowNetwork()
FCFNinstance.loadFCFNfromDisc("smallOneCap")

# Test of alpha individual
alpha = AlphaIndividual(0, FCFNinstance)
alpha.executeAlphaSolver(35)
alpha.relaxedSolver.printModel()
alpha.relaxedSolver.printSolution()
alpha.visualizeAlphaNetwork()

# Test of MILP
FCFNinstance.executeSolver(35)
FCFNinstance.visualizeNetwork()
