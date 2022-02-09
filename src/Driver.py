from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.GraphGenerator import GraphGenerator

"""DRIVER PY FILE"""

# Test of Random Graph Generator
graphGen = GraphGenerator(100, 0.2, 25, 25, 600, 25)
flowNetwork = graphGen.finalizeRandomNetwork("rand1", 1, [20], [50], [1])
# flowNetwork.visualizeNetwork()
population = AlphaPopulation(flowNetwork, 35, 25, 5)
population.evolvePopulation()
flowNetwork.executeSolver(35)

# Test of the FCFN
# flowNetwork = FixedChargeFlowNetwork()
# flowNetwork.loadFCFNfromDisc("small1")

# Test of GA Algo.
# population = AlphaPopulation(flowNetwork, 35, 10, 10)
# population.evolvePopulation()

# Test of MILP
# flowNetwork.executeSolver(35)
# flowNetwork.visualizeNetwork()

# Test of alpha individual
# alpha = AlphaIndividual(0, flowNetwork)
# alpha.executeAlphaSolver(125)
# alpha.visualizeAlphaNetwork()
