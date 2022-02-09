from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# Test of the FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFNfromDisc("small1")

# Test of GA Algo.
population = AlphaPopulation(flowNetwork, 35, 10, 10)
population.evolvePopulation()

# Test of MILP
flowNetwork.executeSolver(35)
flowNetwork.visualizeNetwork()

# Test of alpha individual
# alpha = AlphaIndividual(0, flowNetwork)
# alpha.executeAlphaSolver(125)
# alpha.visualizeAlphaNetwork()
