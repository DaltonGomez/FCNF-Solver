from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# TEST OF RANDOM GRAPH GENERATOR
# graphGen = GraphGenerator(100, 0.05, 10, 10, 50, 20)
# graphGen.finalizeRandomNetwork("r100-5", 1, [20], [50], [1])
# graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("r100-5")

# TEST OF GA POPULATION
population = AlphaPopulation(flowNetwork, 150, 25, 1)
population.evolvePopulation(0.25, 0.05, 0.75)

# TEST OF MILP
flowNetwork.executeSolver(80)
flowNetwork.visualizeNetwork()

# TEST OF ALPHA INDIVIDUAL
# alpha = AlphaIndividual(0, flowNetwork)
# alpha.executeAlphaSolver(125)
# alpha.visualizeAlphaNetwork()
