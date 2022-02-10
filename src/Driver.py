from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork
from src.FixedChargeNetwork.GraphGenerator import GraphGenerator

"""DRIVER PY FILE"""

# TEST OF RANDOM GRAPH GENERATOR
graphGen = GraphGenerator(100, 0.01, 10, 10, 50, 20)
graphGen.finalizeRandomNetwork("r100-1", 3, [50], [20], [1])
graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("r100-1")

# TEST OF GA POPULATION
population = AlphaPopulation(flowNetwork, 80, 25, 50)
population.evolvePopulation(0.25, 0.05, 0.75)

# TEST OF MILP
flowNetwork.executeSolver(80)
flowNetwork.visualizeNetwork()

# TEST OF ALPHA INDIVIDUAL
# alpha = AlphaIndividual(0, flowNetwork)
# alpha.executeAlphaSolver(125)
# alpha.visualizeAlphaNetwork()
