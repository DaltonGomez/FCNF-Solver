from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# TEST OF RANDOM GRAPH GENERATOR
# graphGen = GraphGenerator("r1000-3", 1000, 0.03, 50, 50, [50, 200], [10, 50], [20, 100], [1, 10], [10, 50], 3)
# graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("r100-3")

# TEST OF GA POPULATION
population = AlphaPopulation(flowNetwork, 100, 50, 30)
population.setHyperparameters(0.05, 0.75)
population.evolvePopulation()

# TEST OF MILP
flowNetwork.executeSolver(100)
flowNetwork.visualizeNetwork()

# TEST OF ALPHA INDIVIDUAL
# alphaFN = AlphaIndividual(flowNetwork)
# alphaFN.initializeAlphaValuesRandomlyOnRange(0.10, 0.60)
# alphaFN.executeAlphaSolver(1000)
# alphaFN.visualizeAlphaNetwork(endCatName="1")
# alphaFN.allUsedPaths()
