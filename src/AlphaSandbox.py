from src.AlphaGeneticSolver.AlphaIndividual import AlphaIndividual
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork
from src.FixedChargeNetwork.GraphGenerator import GraphGenerator

"""USED FOR EXPERIMENTATION WITH ALPHA VALUES"""

# TEST OF GRAPH GENERATOR
graphGen = GraphGenerator(100, 0.05, 10, 10, 50, 20)
graphGen.finalizeRandomNetwork("r100-5", 3, [50], [20], [1])
graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("r100-5")

# TEST OF ALPHA INDIVIDUAL
alphaFN = AlphaIndividual(flowNetwork)
alphaFN.executeAlphaSolver(150)
alphaFN.visualizeAlphaNetwork()

# TEST OF MILP
flowNetwork.executeSolver(150)
flowNetwork.visualizeNetwork()

# TEST OF GA POPULATION
# population = AlphaPopulation(flowNetwork, 80, 25, 1)
# population.evolvePopulation(0.25, 0.05, 0.75)
