from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# TEST OF RANDOM GRAPH GENERATOR
# graphGen = GraphGenerator("r100-5", 100, 0.05, 10, 10, [50, 200], [10, 50], [20, 100], [1, 10], [10, 50], 3)
# graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("r100-5")

# TEST OF SOLVERS
# alpha = AlphaIndividual(flowNetwork)
# alpha.executeAlphaSolver(100)
# alpha.visualizeAlphaNetwork()
flowNetwork.executeSolver(80)
flowNetwork.visualizeNetwork()

# TEST OF GA POPULATION
# population = AlphaPopulation(flowNetwork, 80, 25, 50)
# population.evolvePopulation(0.25, 0.05, 0.75)
