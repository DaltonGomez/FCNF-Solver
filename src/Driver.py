from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork
from src.FixedChargeNetwork.GraphGenerator import GraphGenerator

"""DRIVER PY FILE"""

# TEST OF RANDOM GRAPH GENERATOR
graphGen = GraphGenerator(30, 0.3, 5, 5, 50, 20)
graphGen.finalizeRandomNetwork("rand30", 1, [20], [50], [1])
graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("rand30")

# TEST OF GA POPULATION
population = AlphaPopulation(flowNetwork, 100, 25, 5)
population.evolvePopulation(0.25, 0.05, 0.75)
flowNetwork.executeSolver(100)
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
