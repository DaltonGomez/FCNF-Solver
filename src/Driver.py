from src.FixedChargeNetwork.GraphGenerator import GraphGenerator

"""DRIVER PY FILE"""

# Test of the FCFN
# flowNetwork = FixedChargeFlowNetwork()
# flowNetwork.loadFCFNfromDisc("small1")

# Test of GA Algo.
# population = AlphaPopulation(flowNetwork, 35, 10, 10)
# population.evolvePopulation()

# Test of MILP
# flowNetwork.executeSolver(35)
# flowNetwork.visualizeNetwork()

# Test of Random Graph Generator
graphGen = GraphGenerator(500, 0.2, 25, 25, 75, 25)
flowNetwork = graphGen.finalizeRandomNetwork("rand1", 1, [10], [10], [1])
flowNetwork.executeSolver(35)
flowNetwork.visualizeNetwork()

# Test of alpha individual
# alpha = AlphaIndividual(0, flowNetwork)
# alpha.executeAlphaSolver(125)
# alpha.visualizeAlphaNetwork()
