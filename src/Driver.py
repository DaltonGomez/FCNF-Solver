from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# TEST OF RANDOM GRAPH GENERATOR
# graphGen = GraphGenerator(5, 0.2, 1, 1, 50, 10)
# graphGen.finalizeRandomNetwork("rand1", 1, [20], [50], [1])
# graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("small1")

# TEST OF GA POPULATION
population = AlphaPopulation(flowNetwork, 10, 25, 5)
population.evolvePopulation()
flowNetwork.executeSolver(10)
flowNetwork.visualizeNetwork()

# TEST OF GA POPULATION
# population = AlphaPopulation(flowNetwork, 35, 25, 5)
# population.evolvePopulation()
# flowNetwork.executeSolver(35)

# TEST OF MILP
# flowNetwork.executeSolver(35)
# flowNetwork.visualizeNetwork()

# TEST OF ALPHA INDIVIDUAL
# alpha = AlphaIndividual(0, flowNetwork)
# alpha.executeAlphaSolver(125)
# alpha.visualizeAlphaNetwork()
