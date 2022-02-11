from src.AlphaGeneticSolver.AlphaIndividual import AlphaIndividual
from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# TEST OF RANDOM GRAPH GENERATOR
# graphGen = GraphGenerator("r100-5", 100, 0.05, 10, 10, [50, 200], [10, 50], [20, 100], [1, 10], [10, 50], 3)
# graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("r100-3")

# TEST OF ALPHA INDIVIDUAL
alphaFN = AlphaIndividual(flowNetwork)
alphaFN.initializeAlphaValuesConstantly(0.15)
alphaFN.executeAlphaSolver(100)
alphaFN.visualizeAlphaNetwork(endCatName="1")
# alphaFN.allUsedPaths()

# TEST OF MILP
flowNetwork.executeSolver(100)
flowNetwork.visualizeNetwork()

# TEST OF GA POPULATION
population = AlphaPopulation(flowNetwork, 80, 25, 50)
population.setHyperparameters(0.05, 0.75)
population.evolvePopulation()
